"""
Semantic chunking for text documents.

Implements intelligent text chunking that preserves semantic context by analyzing
sentence boundaries and embedding similarities. Uses token-based counting for
accurate chunk sizing.
"""

import re
from typing import List
import numpy as np
import tiktoken

from .models.base import BaseEmbeddingModel


class SemanticChunker:
    """
    Semantic chunking that preserves context and meaning using token-based sizing.

    Uses sentence boundaries and similarity thresholds to create chunks
    that maintain coherent context. Optimized with batch embedding generation.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        max_chunk_size: int = 512,
        min_chunk_size: int = 180,
        similarity_threshold: float = 0.6,
        chunk_overlap: int = 30
    ):
        """
        Initialize semantic chunker.

        Args:
            embedding_model: Embedding model for computing sentence similarities
            max_chunk_size: Maximum tokens per chunk (default: 512)
            min_chunk_size: Minimum tokens before considering a split (default: 180)
            similarity_threshold: Cosine similarity threshold for chunk boundaries
                                 Lower = more splits (0.6 is good for technical docs)
            chunk_overlap: Number of tokens to overlap between consecutive chunks (default: 30)
        """
        self.embedding_model = embedding_model
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.chunk_overlap = chunk_overlap

        # Initialize tiktoken tokenizer (cl100k_base = GPT-3.5/4 tokenizer)
        # Good approximation for most modern language models
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple heuristics.

        Args:
            text: Input text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitter - splits on period/exclamation/question followed by space
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine similarity score (-1 to 1, higher = more similar)
        """
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return float(np.dot(emb1, emb2) / (n1 * n2))

    def chunk_text(self, text: str, verbose: bool = False) -> List[str]:
        """
        Create semantic chunks that preserve context with token-based sizing.

        Uses batch embedding generation to minimize API calls, then analyzes
        semantic similarity between consecutive sentences to determine chunk boundaries.
        Applies overlap between chunks to maintain context.

        Args:
            text: Text to chunk
            verbose: Whether to print progress information

        Returns:
            List of text chunks with overlap
        """
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        # Generate all sentence embeddings in one batch call
        if verbose:
            print(f"  Generating embeddings for {len(sentences)} sentences...")

        try:
            embedding_response = self.embedding_model.get_embeddings(sentences)
            embeddings = [np.array(emb) for emb in embedding_response.embeddings]
        except Exception as e:
            if verbose:
                print(f"  Warning: Batch embedding failed ({e}), falling back to simple chunking")
            return self._fallback_chunk_by_tokens(text)

        # Build chunks by analyzing sentence similarities
        chunks = []
        current_chunk = []
        current_token_count = 0
        prev_embedding = None

        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)
            current_embedding = embeddings[i]

            # Decide whether to start a new chunk
            should_split = False

            # Split if adding this sentence exceeds max size (only if we have content)
            if current_chunk and current_token_count + sentence_tokens > self.max_chunk_size:
                should_split = True

            # Split if we're at/above min size and semantic similarity drops
            elif (prev_embedding is not None and
                  current_token_count >= self.min_chunk_size):
                similarity = self.cosine_similarity(prev_embedding, current_embedding)
                if similarity < self.similarity_threshold:
                    should_split = True

            if should_split and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Apply overlap: keep last sentences that total ~chunk_overlap tokens
                overlap_sentences = []
                overlap_tokens = 0
                for sent in reversed(current_chunk):
                    sent_tokens = self.count_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break

                # Start new chunk with overlap + current sentence
                current_chunk = overlap_sentences + [sentence]
                current_token_count = overlap_tokens + sentence_tokens
            else:
                current_chunk.append(sentence)
                current_token_count += sentence_tokens

            prev_embedding = current_embedding

        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _fallback_chunk_by_tokens(self, text: str) -> List[str]:
        """
        Simple fallback: chunk by token count at sentence boundaries.

        Used when embedding generation fails.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_token_count = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if current_chunk and current_token_count + sentence_tokens > self.max_chunk_size:
                chunks.append(" ".join(current_chunk))

                # Apply overlap
                overlap_sentences = []
                overlap_tokens = 0
                for sent in reversed(current_chunk):
                    sent_tokens = self.count_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break

                current_chunk = overlap_sentences + [sentence]
                current_token_count = overlap_tokens + sentence_tokens
            else:
                current_chunk.append(sentence)
                current_token_count += sentence_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
