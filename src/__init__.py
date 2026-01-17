"""
Multimodal RAG system for violin pedagogy and technical documentation.

This package provides a complete RAG pipeline with:
- Text-grounded image retrieval (image descriptions embedded with text model)
- Hybrid search (semantic + BM25)
- Semantic chunking with token-based sizing
- Multiple model providers (Mistral, Ollama)
- Comprehensive evaluation metrics

Example usage:
    >>> from src import MistralEmbeddingModel, OllamaGenerativeModel
    >>> from src import SemanticChunker, VectorDBManager, HybridSearchReranker
    >>> from src import generate_rag_response, RAGEvaluator

    >>> # Initialize models
    >>> embedding_model = MistralEmbeddingModel()
    >>> generative_model = OllamaGenerativeModel(model_name="llava")

    >>> # Set up chunking and retrieval
    >>> chunker = SemanticChunker(embedding_model)
    >>> vector_db = VectorDBManager()
    >>> searcher = HybridSearchReranker(embedding_model, vector_db)

    >>> # Process documents, search, generate responses
    >>> # ... (see notebooks for complete examples)
"""

# Model providers
from .models.base import (
    BaseEmbeddingModel,
    BaseGenerativeModel,
    EmbeddingResponse,
    GenerationResponse
)

from .models.mistral import (
    MistralEmbeddingModel,
    MistralGenerativeModel
)

from .models.ollama import (
    OllamaEmbeddingModel,
    OllamaGenerativeModel
)

# Document processing
from .document_processing import (
    TextChunk,
    ImageData,
    extract_text_from_page,
    extract_images_from_page,
    generate_image_description,
    process_pdf,
    process_pdf_directory,
    is_within_vertical_bounds,
    is_single_color_image,
    should_keep_image
)

# Chunking
from .chunking import SemanticChunker

# Retrieval
from .retrieval import (
    VectorDBManager,
    HybridSearchReranker,
    simple_tokenize,
    print_search_results
)

# Generation
from .generation import (
    generate_rag_response,
    build_rag_prompt,
    format_text_context_with_budget,
    format_image_context_with_budget,
    format_response_with_sources,
    SYSTEM_PROMPTS,
    MAX_CHUNK_CHARS,
    MAX_IMAGE_DESC_CHARS,
    TOTAL_CONTEXT_BUDGET_CHARS
)

# Evaluation
from .evaluation import (
    RAGEvaluator,
    EvaluationResult,
    extract_json_robust,
    extract_contexts_as_text,
    print_evaluation_summary
)

# Version
__version__ = "0.1.0"

# Public API - what gets imported with "from src import *"
__all__ = [
    # Model providers - Base
    'BaseEmbeddingModel',
    'BaseGenerativeModel',
    'EmbeddingResponse',
    'GenerationResponse',

    # Model providers - Mistral
    'MistralEmbeddingModel',
    'MistralGenerativeModel',

    # Model providers - Ollama
    'OllamaEmbeddingModel',
    'OllamaGenerativeModel',

    # Document processing
    'TextChunk',
    'ImageData',
    'extract_text_from_page',
    'extract_images_from_page',
    'generate_image_description',
    'process_pdf',
    'process_pdf_directory',

    # Chunking
    'SemanticChunker',

    # Retrieval
    'VectorDBManager',
    'HybridSearchReranker',
    'print_search_results',

    # Generation
    'generate_rag_response',
    'build_rag_prompt',
    'format_response_with_sources',

    # Evaluation
    'RAGEvaluator',
    'EvaluationResult',
    'print_evaluation_summary',
]
