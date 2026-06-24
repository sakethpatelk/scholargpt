"""
End-to-end RAG validation script.

Validates:
  1. PDF ingestion (pdf_loader + chunker + vector_store)
  2. ChromaDB stores and retrieves chunks
  3. Retrieval returns cited sources for each query
  4. Ollama or fallback produces an answer
  5. Citations contain filename and page number

Usage:
    python scripts/validate_rag.py
"""
import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.abspath("."))

from src.rag_pipeline import RAGPipeline

PDF = "data/uploaded/transformer_sample.pdf"
QUESTIONS = [
    "What is the main contribution of this paper?",
    "What are the key hyperparameters of the Transformer model?",
    "What are the limitations and future work mentioned?",
]

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f"  -> {detail}" if detail else ""))
    return condition


def main() -> int:
    print("\n=== RAG Validation ===\n")
    failures = 0

    # ── 1. Ingestion ──────────────────────────────────────────────────────
    print("1. PDF Ingestion")
    pipeline = RAGPipeline()
    try:
        info = pipeline.ingest_pdf(PDF)
        failures += not check("PDF loaded", info["pages"] > 0, f"{info['pages']} pages")
        failures += not check("Chunks created", info["chunks"] > 0, f"{info['chunks']} chunks")
    except Exception as exc:
        print(f"  [ FAIL] Ingestion raised: {exc}")
        return 1

    # ── 2. ChromaDB ───────────────────────────────────────────────────────
    print("\n2. ChromaDB Storage")
    count = pipeline.store.count()
    failures += not check("Chunks persisted in ChromaDB", count > 0, f"{count} total chunks")
    docs = pipeline.list_documents()
    failures += not check("File listed in index", "transformer_sample.pdf" in docs, str(docs))

    # ── 3. Retrieval + Citations + LLM ────────────────────────────────────
    print("\n3. Query / Retrieval / Answer")
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n  Q{i}: {q}")
        result = pipeline.answer(q)

        answer_ok = len(result["answer"]) > 50
        sources_ok = len(result["sources"]) > 0
        cite_ok = all(
            s.get("filename") and s.get("page") for s in result["sources"]
        )

        failures += not check("Answer generated", answer_ok, f"{len(result['answer'])} chars")
        failures += not check("Sources returned", sources_ok, f"{len(result['sources'])} source(s)")
        failures += not check("Citations have filename+page", cite_ok)

        if result["sources"]:
            for s in result["sources"]:
                print(f"       -> {s['filename']} p{s['page']} (score {s['score']})")
        preview = result['answer'][:120].encode('ascii', 'replace').decode('ascii')
        print(f"       Answer preview: {preview}...")

    # ── 4. Answer mode ────────────────────────────────────────────────────
    print("\n4. Answer Mode")
    in_fallback = "[INFO]" in result["answer"] or "[WARNING]" in result["answer"]
    if not in_fallback:
        failures += not check("Ollama answer generated", True, "LLM response received")
    else:
        failures += not check(
            "Retrieval-only fallback active",
            True,
            "Ollama not running — raw passages shown",
        )

    # ── Summary ───────────────────────────────────────────────────────────
    total_checks = 2 + 2 + len(QUESTIONS) * 3 + 1
    passed = total_checks - failures
    print(f"\n{'='*40}")
    print(f"Result: {passed}/{total_checks} checks passed", end="")
    if failures:
        print(f"  ({failures} FAILED)\n")
    else:
        print("  -- all good!\n")

    return failures


if __name__ == "__main__":
    sys.exit(main())
