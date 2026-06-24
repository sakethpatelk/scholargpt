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
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploaded"
CHROMA_DIR = DATA_DIR / "chroma"

# Create data dirs on import so every module can assume they exist.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ── LLM backend ────────────────────────────────────────────────────────────
# "auto"      : try Anthropic (if key set) → Ollama (if running) → retrieval-only
# "anthropic" : force Claude API (requires ANTHROPIC_API_KEY)
# "ollama"    : force local Ollama
# "none"      : retrieval-only, never call an LLM
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "auto")

# ── Claude API ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")

# ── Ollama (local LLM) ─────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# llama3.1 is the recommended default; mistral is a smaller alternative.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1")

# ── Embeddings ─────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dim, ~80 MB, fast CPU inference — good default.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ───────────────────────────────────────────────────────────────
# 800 chars ≈ ~200 tokens; 150-char overlap keeps sentence context across boundaries.
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))           # candidates from vector store
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "3"))  # kept after reranking
USE_RERANKER: bool = os.getenv("USE_RERANKER", "true").lower() == "true"
# Cross-encoder reranker: slower but significantly more precise than bi-encoder alone.
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── ChromaDB ───────────────────────────────────────────────────────────────
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "research_papers")
