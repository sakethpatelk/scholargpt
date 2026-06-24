"""
LLM answer generation with a two-tier backend:

  1. Ollama  — local model, free, no key needed (default)
  2. Retrieval-only — honest fallback, shows raw passages, never crashes

Backend selection (LLM_BACKEND=auto by default):
  - "auto"   : use Ollama if reachable, else retrieval-only
  - "ollama" : force Ollama
  - "none"   : skip LLM entirely, always show raw passages

Interview note — why Ollama:
  Runs entirely on-device, no API key, no rate limits, works offline.
  We call /api/chat via httpx (already a transitive dependency of the project).
"""
import logging
from typing import Any

import httpx

from src.config import LLM_BACKEND, OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise academic research assistant.

Rules:
1. Answer ONLY using the provided context. Never use outside knowledge.
2. If the context is insufficient, say so clearly rather than guessing.
3. Cite specific sources in your answer using [Source N] notation.
4. Use academic language. Structure answers with clear paragraphs.
5. Do not repeat the question back to the user."""


# ── Public helpers ─────────────────────────────────────────────────────────

def build_context_block(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered source block."""
    if not chunks:
        return "No relevant context found in uploaded documents."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[Source {i}] {chunk['filename']} -- Page {chunk['page']}"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def get_active_backend() -> str:
    """Return the backend that will be used: 'ollama' or 'none'."""
    return _select_backend()


def generate_answer(
    query: str,
    chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """Route to Ollama if available, otherwise return retrieval-only fallback."""
    context = build_context_block(chunks)
    messages = _build_messages(context, query, chat_history)
    backend = _select_backend()

    if backend == "ollama":
        result = _call_ollama(messages)
        if result:
            return result
        logger.warning("Ollama call failed — falling back to retrieval-only")

    return _fallback_answer(query, chunks)


# ── Backend selection ──────────────────────────────────────────────────────

def _select_backend() -> str:
    """Return 'ollama' or 'none'."""
    if LLM_BACKEND == "none":
        return "none"
    if LLM_BACKEND == "ollama":
        return "ollama"
    # auto: use Ollama if reachable
    return "ollama" if _is_ollama_available() else "none"


def _is_ollama_available() -> bool:
    """Return True if the Ollama server is reachable at OLLAMA_BASE_URL."""
    try:
        httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        return True
    except Exception:
        return False


# ── Backend callers ────────────────────────────────────────────────────────

def _build_messages(
    context: str,
    query: str,
    chat_history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """Assemble the messages list (OpenAI-style chat format)."""
    user_message = (
        f"Context from uploaded research papers:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}\n\n"
        "Answer strictly from the context above. "
        "Use [Source N] citations where relevant."
    )
    messages: list[dict[str, str]] = []
    if chat_history:
        for turn in chat_history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


def _call_ollama(messages: list[dict[str, str]]) -> str | None:
    """
    Call local Ollama via /api/chat (non-streaming).
    Returns text on success, None on any error.
    Used by the evaluation script and validate_rag.py.
    """
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "stream": False,
        }
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=300.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as exc:
        logger.error("Ollama error: %s", exc)
        return None


def stream_ollama_answer(
    query: str,
    chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
):
    """
    Generator that yields answer tokens from Ollama one chunk at a time.
    Used by app.py with st.write_stream() so the UI updates incrementally.
    Falls back to yielding the full fallback string if Ollama is unreachable.
    """
    context = build_context_block(chunks)
    messages = _build_messages(context, query, chat_history)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "stream": True,
    }

    try:
        import json as _json
        with httpx.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=300.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = _json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break
    except Exception as exc:
        logger.error("Ollama streaming error: %s", exc)
        yield _fallback_answer(query, chunks)


# ── Fallback ───────────────────────────────────────────────────────────────

def _fallback_answer(query: str, chunks: list[dict[str, Any]]) -> str:
    """Return raw retrieved passages when Ollama is unavailable."""
    if not chunks:
        return (
            "No relevant information found in the uploaded documents. "
            "Try rephrasing your question or uploading more papers."
        )

    lines = [
        "[INFO] Ollama not running. Showing raw retrieved passages:\n",
        f"**Query:** {query}\n",
    ]
    for i, chunk in enumerate(chunks[:3], 1):
        excerpt = chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else "")
        lines.append(
            f"\n**[Source {i}]** `{chunk['filename']}` -- Page {chunk['page']}\n"
            f"> {excerpt}"
        )
    return "\n".join(lines)
