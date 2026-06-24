"""
Two-stage retrieval: bi-encoder recall → cross-encoder reranking.

Stage 1 (bi-encoder): Embed the query, find top-K candidates by cosine
similarity.  Fast O(1) via HNSW index.  Suffers from "dual-encoder gap":
query and passage embeddings are independent, so subtle relevance signals
can be missed.

Stage 2 (cross-encoder): Re-score each candidate by running query+passage
through a single encoder.  Sees both sides together → much more accurate
relevance judgements.  Expensive (O(K) forward passes), so K must be small.

Skipped if USE_RERANKER=false or if the cross-encoder model fails to load
(e.g. no internet on first run).
"""
import logging
from typing import Any

from src.config import RERANK_TOP_K, RERANKER_MODEL, TOP_K, USE_RERANKER
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Module-level singleton so the model is loaded once per process, not per call.
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None and USE_RERANKER:
        try:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(RERANKER_MODEL)
            logger.info("Cross-encoder reranker loaded: %s", RERANKER_MODEL)
        except Exception as exc:
            logger.warning("Reranker unavailable (%s) — skipping.", exc)
    return _reranker


def retrieve(store: VectorStore, query: str) -> list[dict[str, Any]]:
    """
    Return the most relevant chunks for a query.

    With reranker: TOP_K candidates → reranked → top RERANK_TOP_K returned.
    Without reranker: TOP_K candidates returned as-is.
    """
    candidates = store.query(query, n_results=TOP_K)
    if not candidates:
        return []

    reranker = _get_reranker() if USE_RERANKER else None

    if reranker and len(candidates) > 1:
        pairs = [(query, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)
        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        candidates = candidates[:RERANK_TOP_K]
        logger.debug("Reranked %d → %d chunks", TOP_K, len(candidates))

    return candidates
