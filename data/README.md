# Data Directory

This directory is where you place your own PDF documents for processing.

## About This Implementation

This RAG system was developed and tested using violin pedagogy texts (Ivan Galamian's "Principles of Violin Playing and Teaching"). The domain-specific focus demonstrates the system's ability to handle technical educational content with diagrams and specialized terminology.

**The system is domain-agnostic** - it works with any PDF documents you provide.

## Usage

1. Place your PDF files in this directory
2. Run the ingestion notebook (`notebooks/01_ingestion_demo.ipynb`)
3. The system will create a `processed/` subdirectory containing:
   - Vector database (ChromaDB)
   - Extracted images
   - Metadata files

## Example Queries

See `sample_queries.json` for domain-specific examples. For violin pedagogy:
- "What is the correct thumb position for shifting?"
- "How should I hold the bow for spiccato?"
- "What exercises help with vibrato development?"

## Note on Copyright

PDF files and processed databases are excluded from version control (`.gitignore`). Users must supply their own legally obtained documents.

## Testing Without PDFs

To test the system quickly:
- Use public domain PDFs from Project Gutenberg
- Create a small test PDF with sample content
- See the demo notebooks for expected input/output format
