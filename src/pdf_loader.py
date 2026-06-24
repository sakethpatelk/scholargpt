"""
PDF text extraction using PyMuPDF (fitz).

Returns a list of page dicts so downstream code never touches the PDF directly.
Each dict carries the text and enough metadata to reconstruct citations.
"""
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract per-page text from a PDF file.

    Returns a list of dicts:
        {text, page (1-indexed), filename, filepath}

    Empty pages are skipped.  Raises on unreadable files so the caller
    (rag_pipeline) can surface a user-friendly error instead of silently
    returning nothing.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[dict[str, Any]] = []

    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")  # plain text; "blocks" for layout-aware
            if text.strip():
                pages.append(
                    {
                        "text": text,
                        "page": page_num + 1,        # humans count from 1
                        "filename": pdf_path.name,
                        "filepath": str(pdf_path),
                    }
                )
        doc.close()
    except fitz.FileDataError as exc:
        raise ValueError(f"Cannot parse PDF '{pdf_path.name}': {exc}") from exc

    logger.info("Extracted %d non-empty pages from '%s'", len(pages), pdf_path.name)
    return pages
