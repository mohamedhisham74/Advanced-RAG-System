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

# ── Text cleanup patterns ─────────────────────────────────────────────────────
_MULTI_NL = re.compile(r"\n{3,}")   # 3+ consecutive newlines → 2
_MULTI_SP = re.compile(r"[ \t]+")  # multiple spaces/tabs → single space

# ── English law number patterns ───────────────────────────────────────────────
_LAW_NUM_PATTERN = re.compile(
    r"Law\s+No\.?\s*\d+\s+(?:of|for)\s+(?:the\s+)?(?:year\s+)?\d+"
    r"|Law\s+Number\s+\d+\s+(?:of|for)\s+\d+"
    r"|Act\s+No\.?\s*\d+\s+of\s+\d+",
    re.IGNORECASE,
)

# ── Structured file headers (manually prepared .txt exports) ──────────────────
_DOC_HEADER = re.compile(r"^DOCUMENT:\s*(.+)$", re.MULTILINE)
_LAW_HEADER = re.compile(r"^LAW_NUMBER:\s*(.+)$", re.MULTILINE)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf(file_path: str) -> dict:
    """
    Extract and clean text from a single English PDF file.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        ProcessedDocument dict with keys:
        document_name, law_number, source_file, raw_text, pages.

    Raises:
        FileNotFoundError: If the PDF path does not exist.
        RuntimeError:      If PyMuPDF fails to open the file.
    """
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

    raw_text  = "\n\n".join(pages_text)
    cleaned   = clean_text(raw_text)
    metadata  = extract_document_metadata(cleaned, path.name)

    result = {
        "document_name": metadata["document_name"],
        "law_number":    metadata["law_number"],
        "source_file":   path.name,
        "raw_text":      cleaned,
        "pages":         len(pages_text),
    }

    logger.info("  ✓ %s — %d pages, %d chars", path.name, result["pages"], len(cleaned))
    return result


def load_all_pdfs(raw_dir: str) -> list[dict]:
    """
    Load all PDF files from the raw directory.

    Args:
        raw_dir: Path to data/tax_knowledge_base/raw/

    Returns:
        List of ProcessedDocument dicts.
    """
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    pdf_files = sorted(raw_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", raw_dir)
        return []

    logger.info("Found %d PDF files in %s", len(pdf_files), raw_dir)

    documents: list[dict] = []
    for pdf_path in pdf_files:
        try:
            doc = load_pdf(str(pdf_path))
            documents.append(doc)
        except Exception as exc:
            logger.error("  ✗ Skipping %s: %s", pdf_path.name, exc)

    logger.info("Loaded %d / %d documents successfully", len(documents), len(pdf_files))
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Basic text cleanup: collapse whitespace and excessive blank lines.

    Args:
        text: Raw text extracted from PDF.

    Returns:
        Cleaned text string.
    """
    text = _MULTI_SP.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Metadata extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_document_metadata(text: str, filename: str) -> dict:
    """
    Parse law number and document name from text content or filename.

    Priority:
        1. Structured headers (DOCUMENT: / LAW_NUMBER:)
        2. Law number pattern in first 600 chars
        3. Law number pattern in filename
        4. Filename stem as fallback

    Returns:
        {"document_name": str, "law_number": str}
    """
    document_name = ""
    law_number    = ""

    # 1. Structured headers
    doc_match = _DOC_HEADER.search(text)
    if doc_match:
        document_name = doc_match.group(1).strip()

    law_header_match = _LAW_HEADER.search(text)
    if law_header_match:
        law_number = law_header_match.group(1).strip()

    # 2. Law number from first 600 chars of text
    if not law_number:
        law_match = _LAW_NUM_PATTERN.search(text[:600])
        if law_match:
            law_number = law_match.group(0).strip()

    # 3. Law number from filename
    if not law_number:
        law_match = _LAW_NUM_PATTERN.search(filename)
        if law_match:
            law_number = law_match.group(0).strip()

    # 4. Filename fallback
    if not document_name:
        stem = Path(filename).stem
        stem = re.sub(r"\s*\d+$", "", stem)
        document_name = stem.strip()

    return {"document_name": document_name, "law_number": law_number}


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_processed_text(document: dict, processed_dir: str) -> str:
    """
    Save a ProcessedDocument's cleaned text to processed/ as a .txt file.

    File format:
        DOCUMENT: <document_name>
        LAW_NUMBER: <law_number>
        SOURCE_FILE: <source_file>
        PAGES: <pages>
        ──────────────────────────────────────────────────────────
        <cleaned text body>

    Args:
        document:      ProcessedDocument dict.
        processed_dir: Path to data/tax_knowledge_base/processed/

    Returns:
        Absolute path to the saved .txt file.
    """
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
    logger.info("Saved → %s (%d chars)", output_file.name, len(document["raw_text"]))
    return str(output_file)
