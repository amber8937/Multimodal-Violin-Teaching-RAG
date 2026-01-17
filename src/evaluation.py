"""
RAG evaluation metrics: faithfulness, answer relevancy, and context relevancy.

Implements RAGAS-style metrics with improvements for robustness:
- Per-chunk context budgeting (not raw truncation)
- Evidence-based faithfulness scoring
- Fixed embedding API consistency
- Proper context size limits for technical documents
- Cosine similarity properly mapped to [0,1]

IMPORTANT: Pass RAW RETRIEVER OUTPUTS to evaluation metrics (not budgeted prompt text).
This evaluates retrieval quality, not prompt construction.
"""

import re
import json
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

from .models.base import BaseEmbeddingModel, BaseGenerativeModel


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EvaluationResult:
    """Store evaluation results for a single query."""
    query: str
    answer: str
    contexts: List[str]  # Extracted text from context dicts/strings
    faithfulness_score: float
    answer_relevancy_score: float
    context_relevancy_score: float
    num_contexts: int
    metadata: Optional[Dict] = None


# ============================================================================
# JSON EXTRACTION UTILITIES
# ============================================================================

def extract_json_robust(response_text: str) -> Optional[dict]:
    """
    Extract JSON from LLM response with multiple fallback strategies.

    Handles:
    - Clean JSON
    - Markdown code blocks
    - Truncated JSON (extracts available values)
    - Calculates scores from counts when score is missing
    - Extracts "questions" arrays

    Args:
        response_text: LLM response containing JSON

    Returns:
        Extracted dictionary or None
    """
    if not response_text:
        return None

    text = response_text.strip()

    # Strategy 1: Direct parse (clean JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Remove markdown code blocks
    if "```json" in text:
        try:
            json_content = text.split("```json")[1].split("```")[0].strip()
            return json.loads(json_content)
        except (IndexError, json.JSONDecodeError):
            pass

    if "```" in text:
        try:
            json_content = text.split("```")[1].split("```")[0].strip()
            return json.loads(json_content)
        except (IndexError, json.JSONDecodeError):
            pass

    # Strategy 3: Regex for complete JSON object
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Strategy 4: Extract individual values (for truncated JSON)
    extracted = {}

    # Extract scores directly
    faith_score = re.search(r'"faithfulness_score"\s*:\s*([\d.]+)', text)
    if faith_score:
        extracted['faithfulness_score'] = float(faith_score.group(1))

    ans_score = re.search(r'"answer_relevancy_score"\s*:\s*([\d.]+)', text)
    if ans_score:
        extracted['answer_relevancy_score'] = float(ans_score.group(1))

    ctx_score = re.search(r'"context_relevancy_score"\s*:\s*([\d.]+)', text)
    if ctx_score:
        extracted['context_relevancy_score'] = float(ctx_score.group(1))

    # Extract questions array (for answer relevancy)
    questions_match = re.search(r'"questions"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if questions_match:
        questions_str = questions_match.group(1)
        # Try to extract quoted strings
        questions = re.findall(r'"([^"]+)"', questions_str)
        if questions:
            extracted['questions'] = questions

    # Calculate from counts if missing
    if 'faithfulness_score' not in extracted:
        total_match = re.search(r'"total_claims"\s*:\s*(\d+)', text)
        supported_match = re.search(r'"supported_claims"\s*:\s*(\d+)', text)

        if total_match and supported_match:
            total = int(total_match.group(1))
            supported = int(supported_match.group(1))
            if total > 0:
                extracted['faithfulness_score'] = supported / total
                extracted['total_claims'] = total
                extracted['supported_claims'] = supported

    if 'context_relevancy_score' not in extracted:
        rel_match = re.search(r'"relevant_count"\s*:\s*(\d+)', text)
        partial_match = re.search(r'"partial_count"\s*:\s*(\d+)', text)
        irrel_match = re.search(r'"irrelevant_count"\s*:\s*(\d+)', text)

        if rel_match and partial_match and irrel_match:
            r = int(rel_match.group(1))
            p = int(partial_match.group(1))
            i = int(irrel_match.group(1))
            total = r + p + i

            if total > 0:
                extracted['context_relevancy_score'] = (r + 0.5 * p) / total
                extracted['relevant_count'] = r
                extracted['partial_count'] = p
                extracted['irrelevant_count'] = i

    return extracted if extracted else None


def extract_contexts_as_text(contexts: Union[List[str], List[Dict]]) -> List[str]:
    """
    Extract text content from contexts (handles both str and dict formats).

    Args:
        contexts: List of contexts (strings or dicts with 'content'/'text' keys)

    Returns:
        List of context text strings
    """
    context_texts = []
    for ctx in contexts:
        if isinstance(ctx, str):
            context_texts.append(ctx)
        elif isinstance(ctx, dict):
            text = ctx.get('content') or ctx.get('text', '')
            if text:
                context_texts.append(str(text))

    return [c.strip() for c in context_texts if c and c.strip()]


# ============================================================================
# RAG EVALUATOR
# ============================================================================

class RAGEvaluator:
    """
    Evaluate RAG system quality with three key metrics.

    Metrics:
    - Faithfulness: Is the answer grounded in retrieved contexts?
    - Answer Relevancy: Does the answer address the question?
    - Context Relevancy: Did retrieval bring useful material?

    IMPORTANT: Pass RAW RETRIEVER OUTPUTS (full chunks), not budgeted prompt text.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        generative_model: BaseGenerativeModel
    ):
        """
        Initialize RAG evaluator.

        Args:
            embedding_model: Embedding model for similarity calculations
            generative_model: Generative model for LLM-based judgments
        """
        self.embedding_model = embedding_model
        self.generative_model = generative_model

    # ========================================================================
    # FAITHFULNESS: Is answer grounded in context?
    # ========================================================================

    def calculate_faithfulness(
        self,
        answer: str,
        contexts: Union[List[str], List[Dict]],
        use_llm: bool = True,
        max_contexts: int = 5,
        max_chars_per_context: int = 1000,
        max_claims: int = 8
    ) -> float:
        """
        Measure if answer is grounded in retrieved contexts.

        Improved version:
        - Budgets by keeping top contexts and truncating each
        - Requires model to cite evidence quotes for each claim
        - Limits claims to prevent JSON explosion
        - Robust JSON extraction with fallbacks

        Args:
            answer: Generated answer to check
            contexts: Retrieved contexts (raw retriever outputs, not budgeted)
            use_llm: Whether to use LLM for judgment (True) or embedding fallback (False)
            max_contexts: Maximum number of contexts to include (default: 5)
            max_chars_per_context: Max characters per context chunk (default: 1000)
            max_claims: Maximum claims to extract (prevents JSON overflow, default: 8)

        Returns:
            Faithfulness score (0-1, higher = more faithful)
        """
        if not answer or not contexts:
            return 0.0

        # Extract text from contexts
        context_texts = extract_contexts_as_text(contexts)

        if not context_texts:
            return 0.0

        if use_llm:
            # Keep top N contexts, truncate each to preserve boundaries
            selected_contexts = context_texts[:max_contexts]
            truncated_contexts = [ctx[:max_chars_per_context] for ctx in selected_contexts]

            # Format contexts with clear boundaries
            context_text = "\n---\n".join([
                f"Context {i+1}:\n{ctx}"
                for i, ctx in enumerate(truncated_contexts)
            ])

            # Truncate answer to prevent overflow
            answer_text = answer[:2000]

            prompt = f"""Evaluate if the answer is faithfully grounded in the provided contexts.

**Contexts:**
{context_text}

**Answer to evaluate:**
{answer_text}

**Task:**
1. Break the answer into factual claims (statements that can be verified)
2. Extract at most {max_claims} claims (group similar ones if needed)
3. For each claim, determine if it's supported by the contexts
4. For supported claims, provide a SHORT evidence quote (max 20 words)

**Output JSON only:**
{{
  "total_claims": <number>,
  "supported_claims": <number>,
  "faithfulness_score": <supported/total>,
  "evidence": [
    {{"claim": "...", "supported": true/false, "quote": "short evidence..."}},
    ...
  ]
}}"""

            try:
                response = self.generative_model.generate(
                    prompt=prompt,
                    temperature=0.0,  # Deterministic for evaluation
                    max_tokens=1500
                )

                result = extract_json_robust(response.text)
                if result and 'faithfulness_score' in result:
                    return float(result['faithfulness_score'])

            except Exception as e:
                print(f"Warning: Faithfulness LLM evaluation failed: {e}")

        # Fallback: embedding-based similarity
        return self._faithfulness_embedding_fallback(answer, context_texts[:max_contexts])

    def _faithfulness_embedding_fallback(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Embedding-based faithfulness approximation.

        Maps cosine similarity from [-1, 1] to [0, 1].

        Args:
            answer: Answer text
            contexts: Context texts

        Returns:
            Average similarity score (0-1)
        """
        try:
            # Get embeddings
            all_texts = [answer] + contexts
            response = self.embedding_model.get_embeddings(all_texts)
            embeddings = [np.array(emb) for emb in response.embeddings]

            answer_emb = embeddings[0]
            context_embs = embeddings[1:]

            # Check for zero norms (rare but possible)
            answer_norm = np.linalg.norm(answer_emb)
            if answer_norm == 0:
                return 0.5

            # Calculate similarities and map to [0, 1]
            similarities_01 = []
            for ctx_emb in context_embs:
                ctx_norm = np.linalg.norm(ctx_emb)
                if ctx_norm == 0:
                    continue

                # Cosine similarity in [-1, 1]
                cos_sim = np.dot(answer_emb, ctx_emb) / (answer_norm * ctx_norm)

                # Map to [0, 1]
                score_01 = (cos_sim + 1.0) / 2.0
                similarities_01.append(score_01)

            if not similarities_01:
                return 0.5

            return float(np.mean(similarities_01))

        except Exception as e:
            print(f"Warning: Embedding fallback failed: {e}")
            return 0.5

    # ========================================================================
    # ANSWER RELEVANCY: Does answer address the question?
    # ========================================================================

    def calculate_answer_relevancy(
        self,
        query: str,
        answer: str,
        num_questions: int = 3,
        max_answer_chars: int = 1000
    ) -> float:
        """
        Measure if answer addresses the query.

        Method: Generate questions from answer, compare to original query.

        Improvements:
        - Fixed embedding API consistency
        - Cap answer used for question generation (800-1200 chars)
        - Enforce specific question types
        - Fallback if fewer questions produced
        - Temperature=0.0 for deterministic eval
        - Robust question extraction

        Args:
            query: Original user query
            answer: Generated answer
            num_questions: Number of questions to generate (default: 3)
            max_answer_chars: Max answer chars for question generation (default: 1000)

        Returns:
            Answer relevancy score (0-1, higher = more relevant)
        """
        if not query or not answer:
            return 0.0

        # Truncate answer to prevent overflow and drift
        answer_truncated = answer[:max_answer_chars]

        prompt = f"""Generate {num_questions} specific questions that can be answered by the following text.

**Requirements:**
- Questions should be specific and directly answerable from the text
- Avoid generic phrasing
- Questions should capture the main points
- Output JSON only

**Text:**
{answer_truncated}

**Output JSON:**
{{
  "questions": ["question 1", "question 2", ...]
}}"""

        try:
            response = self.generative_model.generate(
                prompt=prompt,
                temperature=0.0,  # Deterministic for evaluation
                max_tokens=400
            )

            result = extract_json_robust(response.text)

            if result and 'questions' in result:
                questions = result['questions']

                if len(questions) < num_questions:
                    print(f"Warning: Only generated {len(questions)}/{num_questions} questions")

                if not questions:
                    return 0.5  # Fallback score

                # Get embeddings - FIXED API consistency
                all_texts = [query] + questions
                embedding_response = self.embedding_model.get_embeddings(all_texts)
                vecs = embedding_response.embeddings

                query_emb = np.array(vecs[0])
                question_embs = [np.array(vecs[i]) for i in range(1, len(vecs))]

                # Check for zero norm
                query_norm = np.linalg.norm(query_emb)
                if query_norm == 0:
                    return 0.5

                # Calculate similarities and map to [0, 1]
                similarities_01 = []
                for q_emb in question_embs:
                    q_norm = np.linalg.norm(q_emb)
                    if q_norm == 0:
                        continue

                    # Cosine similarity in [-1, 1]
                    cos_sim = np.dot(query_emb, q_emb) / (query_norm * q_norm)

                    # Map to [0, 1]
                    score_01 = (cos_sim + 1.0) / 2.0
                    similarities_01.append(score_01)

                if not similarities_01:
                    return 0.5

                return float(np.mean(similarities_01))

        except Exception as e:
            print(f"Warning: Answer relevancy evaluation failed: {e}")
            return 0.5

    # ========================================================================
    # CONTEXT RELEVANCY: Is retrieved context useful?
    # ========================================================================

    def calculate_context_relevancy(
        self,
        query: str,
        contexts: Union[List[str], List[Dict]],
        use_llm: bool = True,
        max_contexts: int = 8,
        max_chars_per_context: int = 1000,
        total_context_budget: int = 8000
    ) -> float:
        """
        Measure if retrieved contexts are relevant to the query.

        Improvements:
        - Increased per-context char limit (300 → 1000)
        - Truncate per-context before joining (preserves boundaries)
        - Total budget around 8000 chars
        - Handles both List[str] and List[Dict]
        - Maps cosine to [0, 1] in fallback

        Args:
            query: User query
            contexts: Retrieved contexts (raw retriever outputs)
            use_llm: Whether to use LLM for judgment
            max_contexts: Maximum contexts to evaluate (default: 8)
            max_chars_per_context: Max chars per context (default: 1000)
            total_context_budget: Total char budget (default: 8000)

        Returns:
            Context relevancy score (0-1, higher = more relevant)
        """
        if not query or not contexts:
            return 0.0

        # Extract text from contexts
        context_texts = extract_contexts_as_text(contexts)

        if not context_texts:
            return 0.0

        if use_llm:
            # Truncate per-context to preserve boundaries
            selected_contexts = []
            total_chars = 0

            for i, ctx in enumerate(context_texts[:max_contexts]):
                truncated = ctx[:max_chars_per_context]

                # Check if adding this would exceed budget
                if total_chars + len(truncated) > total_context_budget:
                    break

                selected_contexts.append((i + 1, truncated))
                total_chars += len(truncated)

            # Format with clear boundaries
            contexts_formatted = "\n---\n".join([
                f"[Context {i}]:\n{ctx}"
                for i, ctx in selected_contexts
            ])

            prompt = f"""Evaluate if each context is relevant to answering the query.

**Query:**
{query}

**Contexts to evaluate:**
{contexts_formatted}

**Task:**
For each context, classify as:
- RELEVANT: Contains information that directly answers the query
- PARTIAL: Contains related information but not directly answering
- IRRELEVANT: Unrelated to the query

**Output JSON only:**
{{
  "evaluations": [
    {{"context_num": 1, "relevance": "RELEVANT/PARTIAL/IRRELEVANT"}},
    ...
  ],
  "relevant_count": <number>,
  "partial_count": <number>,
  "irrelevant_count": <number>,
  "context_relevancy_score": <(relevant + 0.5*partial) / total>
}}"""

            try:
                response = self.generative_model.generate(
                    prompt=prompt,
                    temperature=0.0,  # Deterministic for evaluation
                    max_tokens=1000
                )

                result = extract_json_robust(response.text)
                if result and 'context_relevancy_score' in result:
                    return float(result['context_relevancy_score'])

            except Exception as e:
                print(f"Warning: Context relevancy LLM evaluation failed: {e}")

        # Fallback: embedding-based similarity
        return self._context_relevancy_embedding_fallback(query, context_texts[:max_contexts])

    def _context_relevancy_embedding_fallback(
        self,
        query: str,
        contexts: List[str]
    ) -> float:
        """
        Embedding-based context relevancy approximation.

        Maps cosine similarity from [-1, 1] to [0, 1].

        Args:
            query: Query text
            contexts: Context texts

        Returns:
            Average similarity score (0-1)
        """
        try:
            # Get embeddings
            all_texts = [query] + contexts
            response = self.embedding_model.get_embeddings(all_texts)
            embeddings = [np.array(emb) for emb in response.embeddings]

            query_emb = embeddings[0]
            context_embs = embeddings[1:]

            # Check for zero norm
            query_norm = np.linalg.norm(query_emb)
            if query_norm == 0:
                return 0.5

            # Calculate similarities and map to [0, 1]
            similarities_01 = []
            for ctx_emb in context_embs:
                ctx_norm = np.linalg.norm(ctx_emb)
                if ctx_norm == 0:
                    continue

                # Cosine similarity in [-1, 1]
                cos_sim = np.dot(query_emb, ctx_emb) / (query_norm * ctx_norm)

                # Map to [0, 1]
                score_01 = (cos_sim + 1.0) / 2.0
                similarities_01.append(score_01)

            if not similarities_01:
                return 0.5

            return float(np.mean(similarities_01))

        except Exception as e:
            print(f"Warning: Embedding fallback failed: {e}")
            return 0.5

    # ========================================================================
    # EVALUATE SINGLE QUERY
    # ========================================================================

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: Union[List[str], List[Dict]],
        use_llm: bool = True
    ) -> EvaluationResult:
        """
        Evaluate a single query-answer-contexts triplet.

        Args:
            query: User query
            answer: Generated answer
            contexts: Retrieved contexts (raw retriever outputs)
            use_llm: Whether to use LLM-based evaluation (vs embedding fallback)

        Returns:
            EvaluationResult with all three scores
        """
        # Extract context texts for storage
        context_texts = extract_contexts_as_text(contexts)

        faithfulness = self.calculate_faithfulness(answer, contexts, use_llm=use_llm)
        answer_relevancy = self.calculate_answer_relevancy(query, answer)
        context_relevancy = self.calculate_context_relevancy(query, contexts, use_llm=use_llm)

        return EvaluationResult(
            query=query,
            answer=answer,
            contexts=context_texts,  # Store extracted text, not str(dict)
            faithfulness_score=faithfulness,
            answer_relevancy_score=answer_relevancy,
            context_relevancy_score=context_relevancy,
            num_contexts=len(context_texts)
        )

    # ========================================================================
    # BATCH EVALUATION
    # ========================================================================

    def evaluate_batch(
        self,
        queries: List[str],
        answers: List[str],
        contexts_list: List[Union[List[str], List[Dict]]],
        use_llm: bool = True
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple query-answer-contexts triplets.

        Args:
            queries: List of user queries
            answers: List of generated answers
            contexts_list: List of retrieved contexts (raw retriever outputs)
            use_llm: Whether to use LLM-based evaluation

        Returns:
            List of EvaluationResults
        """
        if not (len(queries) == len(answers) == len(contexts_list)):
            raise ValueError("Queries, answers, and contexts must have same length")

        results = []
        for i, (query, answer, contexts) in enumerate(zip(queries, answers, contexts_list)):
            print(f"Evaluating {i+1}/{len(queries)}...")
            result = self.evaluate(query, answer, contexts, use_llm=use_llm)
            results.append(result)

        return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_evaluation_summary(results: List[EvaluationResult]):
    """
    Print summary statistics for evaluation results.

    Args:
        results: List of evaluation results
    """
    if not results:
        print("No evaluation results to summarize")
        return

    faithfulness_scores = [r.faithfulness_score for r in results]
    answer_relevancy_scores = [r.answer_relevancy_score for r in results]
    context_relevancy_scores = [r.context_relevancy_score for r in results]

    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Number of queries: {len(results)}")
    print(f"\nFaithfulness:       {np.mean(faithfulness_scores):.3f} (±{np.std(faithfulness_scores):.3f})")
    print(f"Answer Relevancy:   {np.mean(answer_relevancy_scores):.3f} (±{np.std(answer_relevancy_scores):.3f})")
    print(f"Context Relevancy:  {np.mean(context_relevancy_scores):.3f} (±{np.std(context_relevancy_scores):.3f})")
    print(f"{'='*80}\n")
