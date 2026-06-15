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

- **Source citations are essential**: Built `rag_query_with_sources()` 
  to expose retrieved chunks alongside generated answers. This 
  directly caught a hallucination — the model claimed "8 private 
  internal documents from a production company," a detail absent 
  from all 3 retrieved sources. Without source visibility, this 
  fabricated answer would have appeared fully credible.
  

  **Phase 1:**
  - Day 1: Environment, LLM connections, witnessed hallucination
  - Day 2: Embeddings, ChromaDB, document chunking, semantic search
  - Day 3: Full RAG pipeline — retrieval + generation + source citations
  - Day 4: Make it work with MULTIPLE PDFs, add proper metadata, and move core logic into a reusable Python file (src/)

## Author
Built as a portfolio project to demonstrate production RAG systems.
