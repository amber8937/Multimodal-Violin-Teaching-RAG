"""
FastAPI backend for the Multimodal RAG system.

This API provides endpoints for the React UI to interact with the RAG system.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import yaml

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mistral import MistralEmbeddingModel, MistralGenerativeModel
from src.retrieval import VectorDBManager, HybridSearchReranker
from src.generation import generate_rag_response

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal RAG API",
    description="API for violin pedagogy RAG system",
    version="0.1.0"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    includeImages: bool = True


class Source(BaseModel):
    documentName: str
    pageNumber: int
    relevance: float
    excerpt: str
    fullText: Optional[str] = None  # Full chunk text
    imagePath: Optional[str] = None  # Path to extracted image
    isImage: bool = False  # Whether this is an image source


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    message: str
    vectorDBStats: Optional[Dict] = None


# Global variables for models and vector DB
embedding_model = None
generative_model = None
vector_db = None
searcher = None
config = None


def load_config():
    """Load configuration file."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_system():
    """Initialize models and vector database on startup."""
    global embedding_model, generative_model, vector_db, searcher, config

    print("🔧 Initializing RAG system...")

    # Load config
    config = load_config()
    print("✅ Configuration loaded")

    # Check API key
    if not os.getenv('MISTRAL_API_KEY'):
        raise ValueError("MISTRAL_API_KEY not found in environment")
    print("✅ MISTRAL_API_KEY found")

    # Initialize models
    print("🔧 Loading Mistral models...")
    embedding_model = MistralEmbeddingModel(model_name=config['mistral']['embedding_model'])
    generative_model = MistralGenerativeModel(model_name=config['mistral']['generation_model'])
    print("✅ Models loaded")

    # Load vector database
    db_path = Path(__file__).parent.parent / "data" / "processed" / "chroma_db"
    if not db_path.exists():
        raise ValueError(
            f"Vector database not found at {db_path}. "
            "Please run test_system.py first to process your documents."
        )

    print(f"🔧 Loading vector database from {db_path}...")
    vector_db = VectorDBManager(persist_directory=str(db_path))

    # Check if database has data
    text_count = vector_db.text_collection.count()
    image_count = vector_db.image_collection.count()

    if text_count == 0 and image_count == 0:
        raise ValueError(
            "Vector database is empty. Please run test_system.py first to process your documents."
        )

    print(f"✅ Vector database loaded ({text_count} text chunks, {image_count} images)")

    # Initialize searcher
    searcher = HybridSearchReranker(
        embedding_model=embedding_model,
        vector_db=vector_db,
        semantic_weight=config['retrieval']['semantic_weight']
    )
    print("✅ Hybrid searcher initialized")

    print("\n🎻 RAG system ready!\n")


@app.on_event("startup")
async def startup_event():
    """Run initialization on server startup."""
    try:
        initialize_system()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    if vector_db is None:
        return HealthResponse(
            status="error",
            message="System not initialized"
        )

    text_count = vector_db.text_collection.count()
    image_count = vector_db.image_collection.count()

    return HealthResponse(
        status="healthy",
        message="Multimodal RAG API is running",
        vectorDBStats={
            "textChunks": text_count,
            "images": image_count
        }
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return await root()


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint for RAG system.

    Takes a user query and returns an AI-generated answer with sources.
    """
    if searcher is None or generative_model is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please check server logs."
        )

    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    try:
        # Search for relevant context
        print(f"\n📊 Query: '{request.query}'")

        text_results, image_results = searcher.search(
            query=request.query,
            top_k_candidates=config['retrieval']['top_k_candidates'],
            top_k_text=config['retrieval']['top_k_final'],
            top_k_images=config['retrieval']['top_k_images'] if request.includeImages else 0
        )

        def filter_image_results(results, query: str):
            if not results:
                return []

            def score(result):
                value = result.get("score", None)
                if isinstance(value, (int, float)):
                    return float(value)
                return float("-inf")

            best_score = max(score(result) for result in results)
            delta = 0.05
            filtered = [
                result for result in results
                if score(result) >= best_score - delta
            ]
            filtered.sort(key=score, reverse=True)

            q = query.lower()
            intent_keywords = (
                "figure",
                "diagram",
                "image",
                "screenshot",
                "shown",
                "picture",
            )
            if any(keyword in q for keyword in intent_keywords) and not filtered:
                filtered = [max(results, key=score)]

            return filtered

        filtered_image_results = filter_image_results(image_results, request.query)

        print(
            "   Found "
            f"{len(text_results)} text results, "
            f"{len(image_results)} image results "
            f"({len(filtered_image_results)} after gating)"
        )

        # Generate response
        answer = generate_rag_response(
            query=request.query,
            text_results=text_results,
            image_results=filtered_image_results,
            generative_model=generative_model,
            domain="violin_pedagogy",
            include_image_bytes=False,
            include_image_descriptions=request.includeImages,
            max_tokens=2048,  # Increased from default 1024 to prevent cutoff
            temperature=0.2
        )

        print(f"   Generated {len(answer)} character response")

        # Format sources for UI
        sources = []

        # Add text sources
        for result in text_results[:5]:  # Top 5 sources for UI
            metadata = result.get('metadata', {})
            content = result.get('content', '')

            # Extract excerpt (first 200 chars or first sentence)
            excerpt = content[:200]
            if len(content) > 200:
                excerpt += "..."

            sources.append(Source(
                documentName=metadata.get('file_name', 'Unknown'),
                pageNumber=metadata.get('page_num', 0),
                relevance=float(result.get('score', 0.0)),
                excerpt=excerpt,
                fullText=content,  # Send full chunk text
                isImage=False
            ))

        # Add image sources if included
        if request.includeImages:
            for result in filtered_image_results[:3]:  # Top 3 image sources
                metadata = result.get('metadata', {})
                description = result.get('content', '')
                image_path = metadata.get('image_path', '')

                excerpt = description[:200]
                if len(description) > 200:
                    excerpt += "..."

                sources.append(Source(
                    documentName=f"[Image] {metadata.get('file_name', 'Unknown')}",
                    pageNumber=metadata.get('page_num', 0),
                    relevance=float(result.get('score', 0.0)),
                    excerpt=excerpt,
                    fullText=description,  # Full image description
                    imagePath=image_path,  # Path to the extracted image
                    isImage=True
                ))

        return QueryResponse(
            answer=answer,
            sources=sources,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        print(f"❌ Error processing query: {e}")
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/images/{image_filename}")
async def serve_image(image_filename: str):
    """
    Serve extracted images from the processed data directory.
    """
    images_dir = Path(__file__).parent.parent / "data" / "processed" / "images"
    image_path = images_dir / image_filename

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path)


if __name__ == "__main__":
    import uvicorn

    print("\n🎻 Starting Multimodal RAG API server...\n")

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes during development
        log_level="info"
    )
