# from pathlib import Path
# src_path = Path(r"C:\Users\jasme\Documents\rag-knowledge-assistant\src")
# src_path.mkdir(exist_ok=True)
# print(f"✅ src/ folder ready at: {src_path}")
# print(f"Contents: {list(src_path.iterdir())}")

"""
rag_engine.py
Core RAG pipeline: retrieval, prompting, and generation.
Used by notebooks (Phase 1) and FastAPI backend (Phase 2).
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# --- Setup paths ---
PROJECT_ROOT = Path(__file__).parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

# --- Load environment variables ---
load_dotenv(dotenv_path=ENV_PATH, override=True)

# --- Initialize embedding function ---
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# --- Connect to persistent ChromaDB ---
_chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
_collection = _chroma_client.get_collection(
    name="rag_paper",
    embedding_function=_embedding_fn
)

# --- Connect to Groq ---
_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def retrieve_chunks(query: str, n_results: int = 3):
    """Retrieve the most relevant chunks and their metadata for a query."""
    results = _collection.query(query_texts=[query], n_results=n_results)
    return results['documents'][0], results['metadatas'][0]


def build_prompt(query: str, chunks: list[str]) -> str:
    """Build a grounded prompt from retrieved chunks and the user's question."""
    context = "\n\n".join(chunks)
    return f"""Answer the question based only on the context below.
If the context doesn't contain enough information, say "I don't have enough information to answer this."

Context:
{context}

Question: {query}

Answer:"""


def generate_answer(prompt: str) -> str:
    """Send a prompt to Groq's LLaMA model and return the generated answer."""
    response = _groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based strictly on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content


def rag_query(question: str, n_results: int = 3) -> dict:
    """
    Full RAG pipeline: retrieve -> prompt -> generate -> return with sources.
    """
    chunks, metadatas = retrieve_chunks(question, n_results)
    prompt = build_prompt(question, chunks)
    answer = generate_answer(prompt)

    sources = []
    for c, m in zip(chunks, metadatas):
        if m is None:
            m = {}
        sources.append({
            "text": c[:150] + "...",
            "source": m.get("source", "unknown"),
            "page": m.get("page", "?")
        })

    return {"question": question, "answer": answer, "sources": sources}


# Quick self-test when running this file directly
if __name__ == "__main__":
    result = rag_query("What challenges does this paper address with PDFs?")
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print("\nSources:")
    for s in result['sources']:
        print(f"  - {s['source']}, page {s['page']}")