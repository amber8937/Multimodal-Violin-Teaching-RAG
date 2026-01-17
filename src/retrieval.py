"""
Vector database management and hybrid search with text-grounded image retrieval.

Implements ChromaDB storage, hybrid search (semantic + BM25), and text-grounded
image retrieval where image descriptions are embedded in the same vector space
as text queries.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from .models.base import BaseEmbeddingModel
from .document_processing import TextChunk, ImageData


# ============================================================================
# BM25 TOKENIZATION
# ============================================================================

def simple_tokenize(text: str) -> List[str]:
    """
    Tokenizer for BM25 keyword search, optimized for technical documents.

    Handles:
    - Technical terms with hyphens (A320-200, AMM-27-00-00)
    - Part numbers with slashes (P/N, I/O)
    - Mixed alphanumerics
    - Preserves meaningful separators

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens
    """
    # Lowercase for case-insensitive matching
    text = text.lower()

    # Extract tokens: alphanumerics with optional hyphens/slashes
    # Matches: a320-200, amm-27-00-00, p/n, io, vibrato, etc.
    tokens = re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", text)

    # Keep tokens with 2+ characters (filters single chars unless they're meaningful)
    # Could be adjusted to keep single-char if needed for specific domains
    tokens = [t for t in tokens if len(t) >= 2]

    return tokens


# ============================================================================
# VECTOR DATABASE MANAGER
# ============================================================================

class VectorDBManager:
    """
    Manages vector storage using ChromaDB.

    Creates separate collections for text chunks and image descriptions.
    Both use the same embedding model for text-grounded retrieval.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        text_collection_name: str = "text_chunks",
        image_collection_name: str = "image_descriptions",
        distance_metric: str = "cosine"
    ):
        """
        Initialize vector database manager.

        Args:
            persist_directory: Path for ChromaDB persistence
            text_collection_name: Name for text collection
            image_collection_name: Name for image description collection
            distance_metric: Distance metric ('cosine', 'l2', 'ip')
        """
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create collections with specified distance metric
        self.text_collection = self.client.get_or_create_collection(
            name=text_collection_name,
            metadata={"hnsw:space": distance_metric}
        )

        self.image_collection = self.client.get_or_create_collection(
            name=image_collection_name,
            metadata={"hnsw:space": distance_metric}
        )

    def add_text_chunks(
        self,
        chunks: List[TextChunk],
        embeddings: List[List[float]]
    ):
        """
        Add text chunks to vector database.

        Args:
            chunks: List of TextChunk objects
            embeddings: Corresponding embedding vectors
        """
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [{
            "file_name": chunk.file_name,
            "page_num": chunk.page_num,
            **chunk.metadata
        } for chunk in chunks]

        self.text_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def add_images(
        self,
        images: List[ImageData],
        embeddings: List[List[float]]
    ):
        """
        Add image descriptions to vector database.

        IMPORTANT: For text-grounded retrieval, embeddings should be generated
        from image DESCRIPTIONS using the TEXT embedding model (not multimodal).

        Args:
            images: List of ImageData objects
            embeddings: Text embeddings of image descriptions
        """
        if not images or not embeddings:
            return

        if len(images) != len(embeddings):
            raise ValueError(f"Mismatch: {len(images)} images but {len(embeddings)} embeddings")

        ids = [f"{img.file_name}_p{img.page_num}_i{img.img_num}" for img in images]
        documents = [img.description for img in images]
        metadatas = [{
            "file_name": img.file_name,
            "page_num": img.page_num,
            "img_num": img.img_num,
            "image_path": img.image_path,
            **img.metadata
        } for img in images]

        self.image_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def query_texts(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Query text collection.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            ChromaDB query results
        """
        results = self.text_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results

    def query_images(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query image collection.

        For text-grounded retrieval, use the SAME embedding as text queries
        (generated from query text, not multimodal).

        Args:
            query_embedding: Query embedding vector (text embedding)
            top_k: Number of results to return

        Returns:
            ChromaDB query results
        """
        results = self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results

    def get_collection_counts(self) -> Dict[str, int]:
        """
        Get document counts for all collections.

        Returns:
            Dictionary with collection names and counts
        """
        return {
            "text_chunks": self.text_collection.count(),
            "image_descriptions": self.image_collection.count()
        }

    def clear_collections(self):
        """Delete and recreate all collections (WARNING: destroys data)."""
        text_name = self.text_collection.name
        image_name = self.image_collection.name
        text_metadata = self.text_collection.metadata
        image_metadata = self.image_collection.metadata

        self.client.delete_collection(text_name)
        self.client.delete_collection(image_name)

        self.text_collection = self.client.get_or_create_collection(
            name=text_name,
            metadata=text_metadata
        )
        self.image_collection = self.client.get_or_create_collection(
            name=image_name,
            metadata=image_metadata
        )


# ============================================================================
# HYBRID SEARCH RERANKER
# ============================================================================

class HybridSearchReranker:
    """
    Implements hybrid search combining semantic and keyword (BM25) ranking.

    Uses text-grounded image retrieval: image descriptions are embedded with
    the TEXT embedding model, enabling unified search across text and images.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_db: VectorDBManager,
        semantic_weight: float = 0.7
    ):
        """
        Initialize hybrid search reranker.

        Args:
            embedding_model: Text embedding model for queries and documents
            vector_db: Vector database manager
            semantic_weight: Weight for semantic score (1 - weight = BM25 weight)
                            Default: 0.7 (70% semantic, 30% BM25)
        """
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.semantic_weight = semantic_weight

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for query text.

        Used for BOTH text and image retrieval (text-grounded approach).

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        embedding_response = self.embedding_model.get_embeddings([query])
        return embedding_response.embeddings[0]

    def keyword_search(
        self,
        query: str,
        semantic_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Perform BM25 keyword-based scoring.

        Args:
            query: Query text
            semantic_results: Results from semantic search

        Returns:
            Dictionary mapping document IDs to normalized BM25 scores (0-1)
        """
        keyword_scores = {}

        if 'documents' in semantic_results and semantic_results['documents']:
            documents = semantic_results['documents'][0]

            if not documents:
                return keyword_scores

            # Tokenize documents and query
            tokenized_docs = [simple_tokenize(doc) for doc in documents]
            tokenized_query = simple_tokenize(query)

            # Create BM25 scorer
            bm25 = BM25Okapi(tokenized_docs)

            # Score each document
            scores = bm25.get_scores(tokenized_query)

            # Normalize to 0-1 range for fair hybrid scoring
            max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

            # Map normalized scores to document IDs
            for i, doc_id in enumerate(semantic_results['ids'][0]):
                keyword_scores[doc_id] = scores[i] / max_score

        return keyword_scores

    def rerank_results(
        self,
        query: str,
        semantic_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using hybrid scoring (semantic + keyword).

        Uses per-query min/max normalization for semantic scores to handle
        different distance scales from ChromaDB (regardless of metric used).

        Args:
            query: Query text
            semantic_results: Raw results from ChromaDB semantic search

        Returns:
            List of result dictionaries sorted by hybrid score
        """
        keyword_scores = self.keyword_search(query, semantic_results)
        reranked = []

        if 'ids' in semantic_results and semantic_results['ids'] and semantic_results['ids'][0]:
            distances = semantic_results['distances'][0]

            # Per-query normalization: smaller distance = better match
            # Convert to score where 1.0 = best, 0.0 = worst
            min_dist = min(distances)
            max_dist = max(distances)

            for i, result_id in enumerate(semantic_results['ids'][0]):
                # Normalize distance to [0, 1] score
                if max_dist == min_dist:
                    # All distances are the same
                    semantic_score = 1.0
                else:
                    # Invert: smaller distance → higher score
                    semantic_score = 1.0 - (distances[i] - min_dist) / (max_dist - min_dist)

                keyword_score = keyword_scores.get(result_id, 0.0)

                # Hybrid score: weighted combination
                hybrid_score = (
                    self.semantic_weight * semantic_score +
                    (1 - self.semantic_weight) * keyword_score
                )

                result_dict = {
                    'id': result_id,
                    'score': hybrid_score,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score,
                    'distance': distances[i],  # Keep raw distance for debugging
                    'content': semantic_results['documents'][0][i],
                    'metadata': semantic_results['metadatas'][0][i]
                }
                reranked.append(result_dict)

        # Sort by hybrid score (highest first)
        reranked.sort(key=lambda x: x['score'], reverse=True)
        return reranked

    def search(
        self,
        query: str,
        top_k_candidates: int = 40,
        top_k_text: int = 10,
        top_k_images: int = 3,
        include_images: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Perform hybrid search on both text and images.

        TEXT-GROUNDED RETRIEVAL:
        Uses the SAME text embedding for both text and image queries.
        Image descriptions were embedded with the text model during ingestion.

        Args:
            query: Query text
            top_k_candidates: Number of candidates to retrieve before reranking
            top_k_text: Final number of text results after reranking
            top_k_images: Final number of image results after reranking
            include_images: Whether to search images

        Returns:
            Tuple of (text_results, image_results) as lists of dicts
        """
        # Generate ONE embedding for the query (used for both text and images)
        query_embedding = self.get_query_embedding(query)

        # Search text chunks
        text_results_raw = self.vector_db.query_texts(
            query_embedding,
            top_k=top_k_candidates
        )
        text_results = self.rerank_results(query, text_results_raw)[:top_k_text]

        # Search image descriptions using SAME embedding (text-grounded)
        image_results = []
        if include_images:
            image_results_raw = self.vector_db.query_images(
                query_embedding,  # Same embedding as text!
                top_k=top_k_candidates
            )
            image_results = self.rerank_results(query, image_results_raw)[:top_k_images]

        return text_results, image_results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_search_results(
    text_results: List[Dict],
    image_results: List[Dict],
    max_content_length: int = 200
):
    """
    Print search results in a readable format.

    Args:
        text_results: Text search results
        image_results: Image search results
        max_content_length: Maximum content length to display
    """
    print(f"\n{'='*80}")
    print(f"TEXT RESULTS ({len(text_results)})")
    print(f"{'='*80}")

    for i, result in enumerate(text_results, 1):
        print(f"\n[{i}] Score: {result['score']:.3f} "
              f"(Semantic: {result['semantic_score']:.3f}, "
              f"Keyword: {result['keyword_score']:.3f})")
        print(f"Source: {result['metadata'].get('file_name', 'unknown')}, "
              f"Page: {result['metadata'].get('page_num', 'unknown')}")
        content = result['content'][:max_content_length]
        print(f"Content: {content}{'...' if len(result['content']) > max_content_length else ''}")

    if image_results:
        print(f"\n{'='*80}")
        print(f"IMAGE RESULTS ({len(image_results)})")
        print(f"{'='*80}")

        for i, result in enumerate(image_results, 1):
            print(f"\n[{i}] Score: {result['score']:.3f} "
                  f"(Semantic: {result['semantic_score']:.3f}, "
                  f"Keyword: {result['keyword_score']:.3f})")
            print(f"Source: {result['metadata'].get('file_name', 'unknown')}, "
                  f"Page: {result['metadata'].get('page_num', 'unknown')}")
            print(f"Image Path: {result['metadata'].get('image_path', 'unknown')}")
            description = result['content'][:max_content_length]
            print(f"Description: {description}{'...' if len(result['content']) > max_content_length else ''}")
