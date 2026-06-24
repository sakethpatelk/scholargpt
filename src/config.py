"""
Central configuration. All env vars are read here so every other module
imports from one place — easy to test, easy to change.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
_local_data = BASE_DIR / "data"

# Streamlit Cloud mounts source code read-only at /mount/src — fall back to /tmp
try:
    _local_data.mkdir(parents=True, exist_ok=True)
    _probe = _local_data / ".write_probe"
    _probe.touch()
    _probe.unlink()
    DATA_DIR = _local_data
except OSError:
    import tempfile
    DATA_DIR = Path(tempfile.gettempdir()) / "scholargpt"

UPLOAD_DIR = DATA_DIR / "uploaded"
CHROMA_DIR = DATA_DIR / "chroma"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ── LLM backend ────────────────────────────────────────────────────────────
# "auto"   : use Ollama if running, else retrieval-only
# "ollama" : force local Ollama
# "none"   : retrieval-only, never call an LLM
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "auto")

# ── Ollama (local LLM) ─────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# llama3.2:3b is fast on CPU; llama3.1 is higher quality but slower.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# ── Embeddings ─────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dim, ~80 MB, fast CPU inference — good default.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ───────────────────────────────────────────────────────────────
# 800 chars ≈ ~200 tokens; 150-char overlap keeps sentence context across boundaries.
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "3"))
USE_RERANKER: bool = os.getenv("USE_RERANKER", "true").lower() == "true"
# Cross-encoder reranker: slower but significantly more precise than bi-encoder alone.
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── ChromaDB ───────────────────────────────────────────────────────────────
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "research_papers")
