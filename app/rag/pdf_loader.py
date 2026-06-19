"""
PDF Loader — extract and clean text from English legal PDFs.

Input:  PDF files
Output: ProcessedDocument dicts
        { document_name, law_number, source_file, raw_text, pages }

Library: PyMuPDF (fitz) — reliable cross-platform PDF extraction
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

_MULTI_NL = re.compile(r"\n{3,}")
_MULTI_SP = re.compile(r"[ \t]+")

_LAW_NUM_PATTERN = re.compile(
    r"Law\s+No\.?\s*\d+\s+(?:of|for)\s+(?:the\s+)?(?:year\s+)?\d+"
    r"|Law\s+Number\s+\d+\s+(?:of|for)\s+\d+"
    r"|Act\s+No\.?\s*\d+\s+of\s+\d+",
    re.IGNORECASE,
)

_DOC_HEADER = re.compile(r"^DOCUMENT:\s*(.+)$", re.MULTILINE)
_LAW_HEADER = re.compile(r"^LAW_NUMBER:\s*(.+)$", re.MULTILINE)


def load_pdf(file_path: str) -> dict:
    """Extract and clean text from a single PDF. Returns ProcessedDocument dict."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise RuntimeError(f"PyMuPDF failed to open {path.name}: {exc}") from exc

    pages_text: list[str] = []
    for page in doc:
        text = page.get_text("text", sort=True)
        if text.strip():
            pages_text.append(text)
    doc.close()

    cleaned  = clean_text("\n\n".join(pages_text))
    metadata = _extract_metadata(cleaned, path.name)

    logger.info("Loaded %s — %d pages, %d chars", path.name, len(pages_text), len(cleaned))

    return {
        "document_name": metadata["document_name"],
        "law_number":    metadata["law_number"],
        "source_file":   path.name,
        "raw_text":      cleaned,
        "pages":         len(pages_text),
    }


def load_all_pdfs(raw_dir: str) -> list[dict]:
    """Load all PDFs from a directory. Skips unreadable files with a warning."""
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Directory not found: {raw_dir}")

    pdf_files = sorted(raw_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", raw_dir)
        return []

    documents: list[dict] = []
    for pdf_path in pdf_files:
        try:
            documents.append(load_pdf(str(pdf_path)))
        except Exception as exc:
            logger.error("Skipping %s: %s", pdf_path.name, exc)

    logger.info("Loaded %d / %d PDFs", len(documents), len(pdf_files))
    return documents


def clean_text(text: str) -> str:
    text = _MULTI_SP.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _extract_metadata(text: str, filename: str) -> dict:
    """Parse document name and law number from text headers or filename."""
    document_name = ""
    law_number    = ""

    m = _DOC_HEADER.search(text)
    if m:
        document_name = m.group(1).strip()

    m = _LAW_HEADER.search(text)
    if m:
        law_number = m.group(1).strip()

    if not law_number:
        m = _LAW_NUM_PATTERN.search(text[:600])
        if m:
            law_number = m.group(0).strip()

    if not law_number:
        m = _LAW_NUM_PATTERN.search(filename)
        if m:
            law_number = m.group(0).strip()

    if not document_name:
        stem = Path(filename).stem
        document_name = re.sub(r"\s*\d+$", "", stem).strip()

    return {"document_name": document_name, "law_number": law_number}


def save_processed_text(document: dict, processed_dir: str) -> str:
    """Persist a ProcessedDocument's text to processed/ as a .txt file."""
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    output_file = processed_path / f"{Path(document['source_file']).stem}.txt"

    header = (
        f"DOCUMENT: {document['document_name']}\n"
        f"LAW_NUMBER: {document['law_number']}\n"
        f"SOURCE_FILE: {document['source_file']}\n"
        f"PAGES: {document['pages']}\n"
        f"{'─' * 60}\n\n"
    )

    output_file.write_text(header + document["raw_text"], encoding="utf-8")
    return str(output_file)
