# RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions from your own documents using LLaMA 3 and ChromaDB.

## Tech Stack

### Ai Model
- LLM: LLaMA 3.1 via Groq API (free)
- Vector Database: ChromaDB
- Orchestration: LangChain
- Backend: FastAPI (coming in Phase-2)
- Frontend: Streamlit (coming in Phase-3)

## Project Status
Phase 1 - Foundation (In Progress)

## Setup
```bash
pip install -r requirements.txt
```

## Engineering Notes & Findings

- **ChromaDB `.add()` behavior**: Calling `.add()` with an existing 
  ID silently overwrites the entry (acts as upsert) rather than 
  raising an error or creating a duplicate. Verified through testing 
  on Day 2.
- **Hallucination observed**: Two different LLMs (LLaMA 3.1 via Groq, 
  and an earlier model) gave confidently wrong definitions of "RAG" 
  itself when asked without retrieval context. Gemini 2.5 Flash 
  answered correctly. This motivated the entire project.
  
## Author
Built as a portfolio project to demonstrate production RAG systems.
