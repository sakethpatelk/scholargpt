"""
Live end-to-end test using Ollama + llama3.1.

Checks:
  1. Ollama reachable and llama3.1 available
  2. Backend auto-selects to 'ollama'
  3. PDF ingested, chunks stored in ChromaDB
  4. Each question gets a *generated* answer (not fallback raw passages)
  5. Citations (filename + page) appear for every answer

Usage:
    python scripts/test_ollama_live.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import httpx

from src.llm import _is_ollama_available, get_active_backend
from src.rag_pipeline import RAGPipeline
from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL

PDF = "data/uploaded/transformer_sample.pdf"

QUESTIONS = [
    "What is self-attention?",
    "Why is multi-head attention useful?",
    "What problem did Transformers solve over RNNs?",
]

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(label: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    suffix = f"  -> {detail}" if detail else ""
    print(f"  [{tag}] {label}{suffix}")
    return ok


def check_model_available() -> bool:
    """Confirm llama3.1 is in Ollama's model list."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(OLLAMA_MODEL in m for m in models)
    except Exception:
        return False


def is_generated(answer: str) -> bool:
    """Return True if the answer looks LLM-generated, not a raw fallback dump."""
    fallback_markers = ["[INFO] No LLM backend", "[WARNING]"]
    return not any(m in answer for m in fallback_markers)


def main() -> int:
    print("\n=== Ollama Live End-to-End Test ===\n")
    failures = 0

    # ── 1. Connectivity ────────────────────────────────────────────────────
    print("1. Ollama Connectivity")
    failures += not check("Ollama server reachable", _is_ollama_available(),
                          f"{OLLAMA_BASE_URL}")
    failures += not check(f"Model '{OLLAMA_MODEL}' available", check_model_available())

    if failures:
        print("\nOllama is not ready. Make sure `ollama serve` is running "
              f"and `{OLLAMA_MODEL}` is pulled.")
        return failures

    # ── 2. Backend detection ───────────────────────────────────────────────
    print("\n2. Backend Detection")
    backend = get_active_backend()
    failures += not check("Auto-selected backend is 'ollama'",
                          backend == "ollama", f"got '{backend}'")

    # ── 3. PDF ingestion ───────────────────────────────────────────────────
    print("\n3. PDF Ingestion")
    pipeline = RAGPipeline()
    try:
        info = pipeline.ingest_pdf(PDF)
        failures += not check("PDF loaded", info["pages"] > 0,
                              f"{info['pages']} pages")
        failures += not check("Chunks stored in ChromaDB", info["chunks"] > 0,
                              f"{info['chunks']} chunks")
    except Exception as exc:
        print(f"  [{FAIL}] Ingestion failed: {exc}")
        return failures + 1

    # ── 4. Questions ───────────────────────────────────────────────────────
    print("\n4. Questions & Answers (llama3.1 via Ollama)\n")
    for i, q in enumerate(QUESTIONS, 1):
        print(f"  Q{i}: {q}")
        result = pipeline.answer(q)

        generated = is_generated(result["answer"])
        has_sources = len(result["sources"]) > 0
        citations_valid = all(
            s.get("filename") and s.get("page") for s in result["sources"]
        )

        failures += not check("Answer is LLM-generated",
                              generated,
                              f"{len(result['answer'])} chars")
        failures += not check("Sources returned",
                              has_sources,
                              f"{len(result['sources'])} source(s)")
        failures += not check("Citations have filename + page",
                              citations_valid)

        if result["sources"]:
            for s in result["sources"]:
                print(f"       cite: {s['filename']} p{s['page']} "
                      f"(score {s['score']})")

        # Print first 300 chars of the generated answer
        preview = result["answer"][:300].encode("ascii", "replace").decode("ascii")
        print(f"\n       Answer:\n       {preview}...\n")

    # ── Summary ────────────────────────────────────────────────────────────
    total = 2 + 1 + 2 + len(QUESTIONS) * 3
    passed = total - failures
    print("=" * 45)
    if failures:
        print(f"Result: {passed}/{total} checks passed  ({failures} FAILED)")
    else:
        print(f"Result: {passed}/{total} checks passed  -- all good!")
    print()
    return failures


if __name__ == "__main__":
    sys.exit(main())
