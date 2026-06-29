"""
Unit tests for the components that don't require an API key or a GPU.

Run: pytest tests/ -v
"""
from unittest.mock import MagicMock, patch

from src.chunker import chunk_pages
from src.llm import _call_ollama, _fallback_answer, _is_ollama_available, build_context_block


# ── chunker ────────────────────────────────────────────────────────────────

def make_page(text: str, page: int = 1, filename: str = "test.pdf") -> dict:
    return {"text": text, "page": page, "filename": filename, "filepath": f"/tmp/{filename}"}


def test_chunk_pages_produces_chunks():
    pages = [make_page("A" * 1000)]
    chunks = chunk_pages(pages)
    assert len(chunks) >= 1


def test_chunk_pages_assigns_monotonic_ids():
    pages = [make_page("B" * 2000, page=1), make_page("C" * 2000, page=2)]
    chunks = chunk_pages(pages)
    ids = [c["chunk_id"] for c in chunks]
    assert ids == list(range(len(ids))), "chunk_ids must be sequential from 0"


def test_chunk_pages_preserves_metadata():
    pages = [make_page("Hello world. " * 40, page=7, filename="paper.pdf")]
    chunks = chunk_pages(pages)
    for chunk in chunks:
        assert chunk["filename"] == "paper.pdf"
        assert chunk["page"] == 7


def test_chunk_pages_skips_empty_text():
    pages = [make_page("   \n  \t  ")]
    chunks = chunk_pages(pages)
    assert chunks == [], "Whitespace-only pages should produce no chunks"


def test_chunk_pages_multipage_chunk_ids_unique():
    pages = [make_page("Word " * 200, page=i) for i in range(1, 6)]
    chunks = chunk_pages(pages)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "All chunk_ids must be unique"


# ── llm helpers (no API call) ──────────────────────────────────────────────

def test_build_context_block_empty():
    result = build_context_block([])
    assert "No relevant context" in result


def test_build_context_block_includes_source_info():
    chunks = [{"text": "Key finding here", "filename": "study.pdf", "page": 4}]
    result = build_context_block(chunks)
    assert "study.pdf" in result
    assert "4" in result          # page number present
    assert "Key finding here" in result


def test_build_context_block_numbers_sources():
    chunks = [
        {"text": "First chunk", "filename": "a.pdf", "page": 1},
        {"text": "Second chunk", "filename": "b.pdf", "page": 2},
    ]
    result = build_context_block(chunks)
    assert "[Source 1]" in result
    assert "[Source 2]" in result


def test_fallback_answer_no_chunks():
    result = _fallback_answer("What is RAG?", [])
    assert "No relevant information" in result


def test_fallback_answer_includes_filename_and_page():
    chunks = [{"text": "Important finding", "filename": "paper.pdf", "page": 5}]
    result = _fallback_answer("What were the findings?", chunks)
    assert "paper.pdf" in result
    assert "5" in result          # page number present


def test_fallback_answer_warns_about_api():
    result = _fallback_answer("x", [{"text": "y", "filename": "z.pdf", "page": 1}])
    # fallback message uses [INFO] prefix when Ollama is not running
    assert "[INFO]" in result or "unavailable" in result.lower() or "No LLM" in result


# ── Ollama integration ──────────────────────────────────────────────────────

def test_is_ollama_available_when_running():
    with patch("src.llm.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        assert _is_ollama_available() is True


def test_is_ollama_available_when_not_running():
    with patch("src.llm.httpx.get", side_effect=Exception("Connection refused")):
        assert _is_ollama_available() is False


def test_call_ollama_returns_text_on_success():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "Ollama-generated answer"}}
    with patch("src.llm.httpx.post", return_value=mock_resp):
        messages = [{"role": "user", "content": "test question"}]
        result = _call_ollama(messages)
    assert result == "Ollama-generated answer"


def test_call_ollama_returns_none_on_connection_error():
    with patch("src.llm.httpx.post", side_effect=Exception("Connection refused")):
        messages = [{"role": "user", "content": "test question"}]
        result = _call_ollama(messages)
    assert result is None


def test_call_ollama_returns_none_on_bad_response():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("500 Server Error")
    with patch("src.llm.httpx.post", return_value=mock_resp):
        messages = [{"role": "user", "content": "test question"}]
        result = _call_ollama(messages)
    assert result is None
