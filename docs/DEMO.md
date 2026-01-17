# Demo Guide

This guide walks you through running the Multimodal RAG system locally.

## Prerequisites

- Python 3.10+
- Node.js 18+
- Mistral API key ([Get one here](https://console.mistral.ai/))

## Step 1: Clone and Install

```bash
git clone https://github.com/yourusername/MRAG.git
cd MRAG
pip install -r requirements.txt
```

## Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your Mistral API key:
```
MISTRAL_API_KEY=your_key_here
```

## Step 3: Add Documents

Place your PDF documents in the `data/sample_docs/` directory.

## Step 4: Process Documents

```bash
# This will extract text/images and create embeddings
python -m src.document_processing
```

## Step 5: Start Backend

```bash
cd backend
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

## Step 6: Start Frontend

Open a new terminal:

```bash
cd UI
npm install
npm run dev
```

The UI will open at `http://localhost:5173`

## Usage

1. Type a question in the search bar
2. View retrieved context (text + images)
3. See the generated answer with source citations
4. Click on citations to view source documents

## Example Queries

Try these sample questions:
- "What are the main topics covered?"
- "Show me diagrams related to [your topic]"
- "Explain [specific concept] from the documents"

## Troubleshooting

**ChromaDB errors**: Delete `chroma_db/` and reprocess documents
**API errors**: Check your `.env` file has the correct Mistral API key
**Frontend not loading**: Ensure backend is running on port 8000
