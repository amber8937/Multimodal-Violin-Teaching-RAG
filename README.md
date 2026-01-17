# 🎻 Multimodal RAG System

A violin teaching Retrieval-Augmented Generation system with multimodal support 
for technical documentation.


https://github.com/user-attachments/assets/f8bfeb73-3f18-4973-9451-e63729fb3f1e


## ✨ Features

- **Multimodal Processing**: Extracts and understands images from PDFs
- **Hybrid Search**: Combines semantic (dense) and keyword (BM25) retrieval
- **Semantic Chunking**: Preserves context using similarity-based boundaries
- **Text-Grounded Image Retrieval**: Unified search across text and images
- **Modern UI**: React frontend with source citations

## 🏗️ Architecture

```
PDF → Text Extraction + Image Extraction
         ↓                    ↓
    Semantic Chunking    Pixtral Description
         ↓                    ↓
    Mistral Embed        Mistral Embed
         ↓                    ↓
         └──── ChromaDB ──────┘
                   ↓
              Hybrid Search
                   ↓
           Pixtral Generation
                   ↓
              React UI
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Mistral API key

### Installation
```bash
git clone https://github.com/yourusername/multimodal-rag.git
cd multimodal-rag
pip install -r requirements.txt
cp .env.example .env  # Add your API key
```

### Process Documents
```bash
python scripts/test_system.py
```

### Run the Application
```bash
# Terminal 1: Backend
cd backend && uvicorn api:app --reload

# Terminal 2: Frontend
cd UI && npm install && npm run dev
```

## 📊 Evaluation Results

| Metric | Score |
|--------|-------|
| Faithfulness | 0.XX |
| Answer Relevancy | 0.XX |
| Context Relevancy | 0.XX |

## 🛠️ Tech Stack

- **Embedding**: Mistral Embed
- **Generation**: Pixtral Large (multimodal)
- **Vector DB**: ChromaDB
- **Backend**: FastAPI
- **Frontend**: React + TypeScript + Tailwind

## 📝 License

MIT
