"""
ChromaDB vector store wrapper.

Design decisions worth explaining in an interview:
- PersistentClient: writes to disk so the index survives restarts.
- cosine space: normalises vector length, so short/long chunks are comparable.
- SentenceTransformer encodes locally — no round-trip latency or API cost for
  embedding, and the same model is used at ingest and query time (consistency).
- upsert: idempotent; re-ingesting the same file just overwrites old vectors.
"""
import logging
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_COLLECTION, CHROMA_DIR, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        # Lazy singleton: loaded once per process, shared across all calls.
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("VectorStore ready — collection '%s'", CHROMA_COLLECTION)

    # ── Write ──────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Embed chunks and upsert into ChromaDB. Returns number stored."""
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        # show_progress_bar gives visibility during large ingests
        embeddings = self.encoder.encode(texts, show_progress_bar=True).tolist()

        self.collection.upsert(
            ids=[str(c["chunk_id"]) for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "page": c["page"],
                    "filename": c["filename"],
                    "filepath": c["filepath"],
                }
                for c in chunks
            ],
        )
        logger.info("Upserted %d chunks into vector store", len(chunks))
        return len(chunks)

    def delete_by_filename(self, filename: str) -> int:
        """Remove all chunks that belong to a specific file."""
        results = self.collection.get(where={"filename": filename})
        if not results["ids"]:
            return 0
        self.collection.delete(ids=results["ids"])
        logger.info("Deleted %d chunks for '%s'", len(results["ids"]), filename)
        return len(results["ids"])

    # ── Read ───────────────────────────────────────────────────────────────

    def query(self, query_text: str, n_results: int = 5) -> list[dict[str, Any]]:
        """
        Embed the query and return the top-n most similar chunks.

        Returns list of dicts: {text, score (cosine sim), page, filename, filepath}
        """
        total = self.collection.count()
        if total == 0:
            return []

        query_embedding = self.encoder.encode([query_text]).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, total),
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[dict[str, Any]] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append(
                {
                    "text": doc,
                    "score": round(1.0 - dist, 4),  # cosine distance → similarity
                    "page": meta["page"],
                    "filename": meta["filename"],
                    "filepath": meta["filepath"],
                }
            )
        return retrieved

    def list_files(self) -> list[str]:
        """Return sorted list of unique filenames in the collection."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        filenames = sorted({m["filename"] for m in results["metadatas"]})
        return filenames

    def count(self) -> int:
        return self.collection.count()
