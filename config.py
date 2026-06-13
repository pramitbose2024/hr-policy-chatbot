import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Provider ──────────────────────────────────────────────
LLM_PROVIDER = "groq"          # "groq" or "openai"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GROQ_MODEL = "llama-3.1-8b-instant"
# OPENAI_MODEL = "gpt-4o-mini"

# ── Embeddings ────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Paths ─────────────────────────────────────────────────────
DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"

# ── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ── Retrieval ─────────────────────────────────────────────────
RETRIEVER_K = 6