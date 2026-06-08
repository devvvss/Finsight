# FinSight — Financial Document Intelligence

A RAG-based financial report analyst that lets you upload PDF financial documents and ask context-aware questions.

## Features
- Upload any financial PDF (annual reports, 10-Ks, cost sheets)
- Semantic search using ChromaDB and nomic-embed-text embeddings
- Local LLM answering via Mistral (no API costs)
- Auto chart generation from numerical answers
- Source page citations on every answer
- Clean dark UI

## Tech Stack
- LangChain
- ChromaDB
- Ollama (Mistral + nomic-embed-text)
- Streamlit
- pdfplumber + PyMuPDF

## Setup
1. Install Ollama from ollama.com
2. Run `ollama pull mistral` and `ollama pull nomic-embed-text`
3. Create a virtual environment with Python 3.11
4. Run `pip install -r requirements.txt`
5. Run `streamlit run app.py`
