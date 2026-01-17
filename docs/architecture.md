# System Architecture

## Overview

This is a production-ready Multimodal RAG (Retrieval-Augmented Generation) system that combines text and image understanding for technical document Q&A.

## Architecture Diagram

```
┌─────────────┐
│   PDF Docs  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│  Document Processing Pipeline   │
├─────────────┬───────────────────┤
│ Text Extract│  Image Extract    │
│     │       │       │           │
│     ▼       │       ▼           │
│  Semantic   │   Pixtral         │
│  Chunking   │   Description     │
└─────┬───────┴───────┬───────────┘
      │               │
      ▼               ▼
┌────────────────────────────────┐
│      Mistral Embed Model       │
└────────────┬───────────────────┘
             │
             ▼
      ┌─────────────┐
      │  ChromaDB   │
      │ (Vector DB) │
      └──────┬──────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌─────────┐     ┌──────────┐
│ BM25    │     │ Semantic │
│ Search  │     │  Search  │
└────┬────┘     └─────┬────┘
     │                │
     └────────┬───────┘
              │
              ▼
       ┌─────────────┐
       │   Reranker  │
       └──────┬──────┘
              │
              ▼
     ┌────────────────┐
     │ Pixtral Large  │
     │  (Generation)  │
     └────────┬───────┘
              │
              ▼
      ┌──────────────┐
      │  React UI    │
      └──────────────┘
```

## Component Details

### 1. Document Processing (`src/document_processing.py`)

**Text Extraction:**
- Uses PyMuPDF to extract text from PDFs
- Preserves structure and formatting

**Image Extraction:**
- Extracts embedded images from PDFs
- Generates descriptions using Pixtral vision model
- Links images to their surrounding text context

### 2. Chunking Strategy (`src/chunking.py`)

**Semantic Chunking:**
- Splits documents based on semantic similarity
- Uses sentence embeddings to find natural breakpoints
- Preserves context within chunks
- Default chunk size: 512 tokens with 50 token overlap

### 3. Embedding (`src/retrieval.py`)

**Model:** `mistral-embed`
- Dimension: 1024
- Embeds both text chunks and image descriptions
- Unified vector space for multimodal retrieval

### 4. Vector Database

**ChromaDB:**
- Stores text and image embeddings
- Metadata includes: source file, page number, chunk type
- Supports hybrid search (semantic + keyword)

### 5. Retrieval (`src/retrieval.py`)

**Hybrid Search:**
1. **Semantic Search**: Dense retrieval using cosine similarity
2. **BM25**: Sparse keyword-based retrieval
3. **Fusion**: Combines results using reciprocal rank fusion

**Retrieval Parameters:**
- Top-k: 5 chunks
- Similarity threshold: 0.3

### 6. Generation (`src/generation.py`)

**Model:** `pixtral-large-latest`
- Multimodal LLM (text + images)
- Context window: 128k tokens
- Retrieves relevant images alongside text
- Generates grounded answers with citations

### 7. API Backend (`backend/api.py`)

**FastAPI Server:**
- `/query` endpoint: Accepts questions, returns answers with sources
- `/health` endpoint: Health check
- CORS enabled for local development

### 8. Frontend (`UI/`)

**React + TypeScript + Tailwind:**
- Clean, responsive interface
- Displays retrieved context (text + images)
- Shows source citations with page numbers
- Real-time loading states

## Design Decisions

### Why Pixtral?
- Native multimodal capabilities (text + vision)
- Supports large context windows
- Strong performance on technical documents

### Why ChromaDB?
- Lightweight, embeddable vector database
- Fast similarity search
- Easy to deploy and maintain

### Why Semantic Chunking?
- Preserves context better than fixed-size chunks
- Reduces split sentences/paragraphs
- Improves retrieval quality

### Why Hybrid Search?
- Semantic search finds conceptually similar content
- BM25 catches exact keyword matches
- Combining both improves recall

## Performance Considerations

- **Embedding caching**: Avoids re-embedding unchanged documents
- **Batch processing**: Processes multiple chunks in parallel
- **Lazy loading**: Images loaded on-demand in UI
- **Vector indexing**: ChromaDB uses HNSW for fast approximate search

## Future Improvements

- [ ] Add reranking model (e.g., Cohere rerank)
- [ ] Implement query expansion
- [ ] Add evaluation metrics dashboard
- [ ] Support more document types (DOCX, HTML)
- [ ] Add user feedback loop for continuous improvement
