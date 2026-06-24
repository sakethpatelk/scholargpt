"""
Text chunking with metadata preservation.

RecursiveCharacterTextSplitter tries larger separators first (double-newline,
single-newline, sentence boundary) before falling back to raw character splits.
This keeps semantically related sentences together, which improves embedding
quality compared to a fixed-size window.
"""
import logging
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def chunk_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Split each page's text into overlapping chunks, forwarding metadata.

    Each output dict:
        {text, chunk_id (global), page, filename, filepath}

    chunk_id is a monotonically increasing integer across all pages so
    ChromaDB has a stable, unique ID for upserts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Preference order: paragraph > line > sentence > word > char
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    chunk_id = 0

    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            if not split.strip():
                continue
            chunks.append(
                {
                    "text": split,
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "filename": page["filename"],
                    "filepath": page["filepath"],
                }
            )
            chunk_id += 1

    logger.info(
        "Created %d chunks from %d pages (chunk_size=%d, overlap=%d)",
        len(chunks),
        len(pages),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    return chunks
