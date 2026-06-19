"""
PDF Loader — English PDF text extraction and cleaning.

Responsibility:
    - Extract text from English PDFs using PyMuPDF (fitz)
    - Clean whitespace and normalize line breaks
    - Parse document metadata from filename and content headers
    - Save processed text to data/tax_knowledge_base/processed/

Input:
    PDF files from data/tax_knowledge_base/raw/

Output:
    List of ProcessedDocument dicts:
    {
        "document_name": str,   # e.g. "Income Tax Law"
        "law_number":    str,   # e.g. "Law No. 91 of 2005"
        "source_file":   str,   # original PDF filename
        "raw_text":      str,   # full extracted + cleaned text
        "pages":         int,   # page count
    }

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
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info("Loading PDF: %s", path.name)

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise RuntimeError(f"PyMuPDF failed to open {path.name}: {exc}") from exc

    pages_text: list[str] = []

    for page in doc:
        page_text = page.get_text("text", sort=True)
        if page_text.strip():
            pages_text.append(page_text)

    doc.close()

    raw_text = "\n\n".join(pages_text)
    cleaned  = clean_text(raw_text)
    metadata = extract_document_metadata(cleaned, path.name)

    return {
        "document_name": metadata["document_name"],
        "law_number":    metadata["law_number"],
        "source_file":   path.name,
        "raw_text":      cleaned,
        "pages":         len(pages_text),
    }


def load_all_pdfs(raw_dir: str) -> list[dict]:
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

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

    return documents


def clean_text(text: str) -> str:
    text = _MULTI_SP.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def extract_document_metadata(text: str, filename: str) -> dict:
    document_name = ""
    law_number    = ""

    doc_match = _DOC_HEADER.search(text)
    if doc_match:
        document_name = doc_match.group(1).strip()

    law_header_match = _LAW_HEADER.search(text)
    if law_header_match:
        law_number = law_header_match.group(1).strip()

    if not law_number:
        law_match = _LAW_NUM_PATTERN.search(text[:600])
        if law_match:
            law_number = law_match.group(0).strip()

    if not law_number:
        law_match = _LAW_NUM_PATTERN.search(filename)
        if law_match:
            law_number = law_match.group(0).strip()

    if not document_name:
        stem = Path(filename).stem
        stem = re.sub(r"\s*\d+$", "", stem)
        document_name = stem.strip()

    return {"document_name": document_name, "law_number": law_number}


def save_processed_text(document: dict, processed_dir: str) -> str:
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    stem        = Path(document["source_file"]).stem
    output_file = processed_path / f"{stem}.txt"

    header = (
        f"DOCUMENT: {document['document_name']}\n"
        f"LAW_NUMBER: {document['law_number']}\n"
        f"SOURCE_FILE: {document['source_file']}\n"
        f"PAGES: {document['pages']}\n"
        f"{'─' * 60}\n\n"
    )

    output_file.write_text(header + document["raw_text"], encoding="utf-8")
    return str(output_file)
