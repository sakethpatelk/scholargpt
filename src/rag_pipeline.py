"""
RAGPipeline: the single public interface for the whole system.

app.py and evaluation.py only import this class; they never touch the
individual modules directly.  This is the "facade" pattern — one entry
point hides the ingest/retrieve/generate complexity.
"""
import logging
from pathlib import Path
from typing import Any

from src.chunker import chunk_pages
from src.llm import generate_answer
from src.pdf_loader import extract_pages
from src.retriever import retrieve
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates PDF ingest → vector storage → retrieval → answer generation."""

    def __init__(self) -> None:
        self.store = VectorStore()

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: str | Path) -> dict[str, Any]:
        """
        Load a PDF, chunk it, embed it, and store it in ChromaDB.

        Re-ingesting the same filename deletes old chunks first so the index
        stays consistent (no duplicate passages from the same file).

        Returns a summary dict for display in the UI.
        """
        pdf_path = Path(pdf_path)
        filename = pdf_path.name

        removed = self.store.delete_by_filename(filename)
        if removed:
            logger.info("Replaced %d stale chunks for '%s'", removed, filename)

        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages)
        count = self.store.add_chunks(chunks)

        return {"filename": filename, "pages": len(pages), "chunks": count}

    # ── Query ──────────────────────────────────────────────────────────────

    def answer(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Run the full RAG pipeline for a user question.

        Returns:
            {answer, sources: [{filename, page, excerpt, score}], query}
        """
        if self.store.count() == 0:
            return {
                "answer": (
                    "No documents have been uploaded yet. "
                    "Please upload one or more PDFs using the sidebar."
                ),
                "sources": [],
                "query": query,
            }

        chunks = retrieve(self.store, query)
        answer_text = generate_answer(query, chunks, chat_history)

        sources = [
            {
                "filename": c["filename"],
                "page": c["page"],
                "excerpt": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
                "score": round(c.get("rerank_score", c.get("score", 0.0)), 3),
            }
            for c in chunks
        ]

        return {"answer": answer_text, "sources": sources, "query": query}

    # ── Management ─────────────────────────────────────────────────────────

    def list_documents(self) -> list[str]:
        return self.store.list_files()

    def delete_document(self, filename: str) -> int:
        return self.store.delete_by_filename(filename)
