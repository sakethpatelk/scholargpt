"""
Offline evaluation script.

Usage (after uploading PDFs via the UI or ingest_pdf directly):

    python -m src.evaluation

Measures per-question latency, source count, and answer length — a simple
proxy for whether the pipeline is retrieving and generating correctly.
A production setup would use RAGAS or a human-labeled golden set.
"""
import json
import logging
import time
from pathlib import Path

from src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.WARNING)

DEFAULT_QUESTIONS = [
    "What is the main research contribution of this paper?",
    "What methodology or experimental setup was used?",
    "What are the key quantitative findings or results?",
    "What limitations do the authors acknowledge?",
    "What future work or open problems are mentioned?",
]


def evaluate(
    questions: list[str] | None = None,
    output_path: str | Path | None = None,
) -> list[dict]:
    """
    Run each question through the RAG pipeline and collect metrics.

    Args:
        questions: Custom question list; defaults to DEFAULT_QUESTIONS.
        output_path: If provided, write results as JSON to this path.

    Returns:
        List of per-question result dicts.
    """
    pipeline = RAGPipeline()

    if not pipeline.list_documents():
        print("⚠️  No documents in the store.  Upload PDFs via the UI first.")
        return []

    questions = questions or DEFAULT_QUESTIONS
    results: list[dict] = []

    print(f"Evaluating {len(questions)} questions across {pipeline.list_documents()} ...\n")

    for q in questions:
        t0 = time.perf_counter()
        result = pipeline.answer(q)
        elapsed = time.perf_counter() - t0

        record = {
            "question": q,
            "answer_length_chars": len(result["answer"]),
            "num_sources": len(result["sources"]),
            "latency_s": round(elapsed, 2),
            "sources": [f"{s['filename']}:p{s['page']}" for s in result["sources"]],
            "answer_preview": result["answer"][:250],
        }
        results.append(record)

        print(f"Q: {q}")
        print(f"   Latency: {elapsed:.2f}s | Sources: {record['sources']}")
        print(f"   Preview: {record['answer_preview'][:120]}…\n")

    # Summary
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    avg_sources = sum(r["num_sources"] for r in results) / len(results)
    print("─" * 60)
    print(f"Questions : {len(results)}")
    print(f"Avg latency : {avg_latency:.2f}s")
    print(f"Avg sources : {avg_sources:.1f}")

    if output_path:
        Path(output_path).write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {output_path}")

    return results


if __name__ == "__main__":
    evaluate()
