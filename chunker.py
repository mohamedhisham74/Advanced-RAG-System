"""
Chunker — Hierarchical Semantic Chunking for English Legal Documents.

Strategy (3 phases per document):

    Phase 1 — Structural Parsing (Rule-based, free)
        Regex detects document hierarchy: Part → Chapter → Section → Article
        Each Article becomes a candidate chunk with inherited metadata.

    Phase 2 — Size Gate
        If candidate <= MAX_CHUNK_CHARS  →  keep as single chunk (no LLM)
        If candidate >  MAX_CHUNK_CHARS  →  send to Phase 3

    Phase 3 — LLM Semantic Split (GPT-4o-mini, cheap)
        Sends the oversized article to GPT-4o-mini.
        LLM identifies semantic boundaries within the article.
        Returns 2-4 sub-chunks that each cover one legal concept.
        Fallback: paragraph split if LLM response is malformed.

Model used: llm_fast (GPT-4o-mini) — cheap, fast, sufficient for splitting
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agents.shared.llm_client import llm_fast

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
MAX_CHUNK_CHARS = 2000   # above this triggers LLM split
OVERLAP_CHARS   = 150    # overlap appended between paragraph sub-chunks

# ── Structural regex patterns (English legal conventions) ─────────────────────
# Article (1) | Article 1 | Article One | Article 1 bis
_ARTICLE_SPLIT_RE = re.compile(
    r"(\n\s*Article\s+(?:\(\s*\d+\s*\)|\d+(?:\s+(?:bis|ter))?|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))",
    re.IGNORECASE,
)

# Chapter One | Chapter 1 | Chapter First
_CHAPTER_RE = re.compile(
    r"^\s*(Chapter\s+(?:[A-Z][a-z]+|\d+)(?:\s+\w+){0,4})",
    re.MULTILINE | re.IGNORECASE,
)

# Section One | Section 1
_SECTION_RE = re.compile(
    r"^\s*(Section\s+(?:[A-Z][a-z]+|\d+)(?:\s+\w+){0,4})",
    re.MULTILINE | re.IGNORECASE,
)

# Part One | Part 1
_PART_RE = re.compile(
    r"^\s*(Part\s+(?:[A-Z][a-z]+|\d+)(?:\s+\w+){0,4})",
    re.MULTILINE | re.IGNORECASE,
)

# Article label extraction — for citation in chunk metadata
_ARTICLE_LABEL_RE = re.compile(
    r"Article\s+(?:\(\s*(\d+)\s*\)|(\d+)|([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))",
    re.IGNORECASE,
)

# ── LLM prompts ────────────────────────────────────────────────────────────────
_SPLIT_SYSTEM = """You are an expert legal document analyzer specializing in tax law.
Your task: split the provided legal article text into semantically coherent segments.
Rules:
- Do NOT modify any word in the text — copy verbatim
- Each segment must cover one complete legal concept or sub-provision
- Number of segments: between 2 and 4 only
- Return JSON only, no additional commentary"""

_SPLIT_HUMAN = """Split the following legal article into semantic segments.
Return: {{"segments": ["First segment...", "Second segment...", ...]}}

Article text:
{article_text}"""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def chunk_all_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk all ProcessedDocuments and return a flat list of chunks.

    Args:
        documents: List of ProcessedDocument dicts from pdf_loader.

    Returns:
        Flat list of Chunk dicts across all documents.
    """
    all_chunks: list[dict] = []

    for document in documents:
        logger.info("Chunking: %s", document["document_name"] or document["source_file"])
        doc_chunks = await chunk_document(document)
        all_chunks.extend(doc_chunks)
        logger.info("  → %d chunks", len(doc_chunks))

    logger.info("Total chunks across all documents: %d", len(all_chunks))
    return all_chunks


async def chunk_document(document: dict) -> list[dict]:
    """
    Split a single ProcessedDocument into semantic chunks.

    Phase 1: Structural split by Article markers.
    Phase 2: Size check per article candidate.
    Phase 3: LLM semantic split for oversized articles.

    Returns:
        List of Chunk dicts:
        {
            "chunk_id":      str,   # "{source_stem}::{index:04d}"
            "document_name": str,
            "law_number":    str,
            "article":       str,   # "Article 3" — used for citation
            "section":       str,   # current chapter/section title
            "chunk_text":    str,
            "char_count":    int,
        }
    """
    text          = document.get("raw_text", "")
    document_name = document.get("document_name", "")
    law_number    = document.get("law_number", "")
    source_stem   = Path(document.get("source_file", "doc")).stem

    if not text.strip():
        logger.warning("Empty text for document: %s", document_name)
        return []

    # ── Phase 1: structural split ──────────────────────────────────────────
    article_segments = _split_by_articles(text)

    # ── Phase 2 + 3: size check → LLM split if needed ─────────────────────
    chunks: list[dict] = []
    chunk_index     = 0
    current_section = ""

    for segment in article_segments:
        seg_text = segment.strip()
        if not seg_text:
            continue

        # Track chapter / section / part context
        section_match = (
            _CHAPTER_RE.search(seg_text)
            or _SECTION_RE.search(seg_text)
            or _PART_RE.search(seg_text)
        )
        if section_match:
            current_section = section_match.group(1).strip()

        article_label = extract_article_number(seg_text)

        if len(seg_text) <= MAX_CHUNK_CHARS:
            chunks.append(_build_chunk(
                chunk_id      = f"{source_stem}::{chunk_index:04d}",
                chunk_text    = seg_text,
                document_name = document_name,
                law_number    = law_number,
                article       = article_label,
                section       = current_section,
            ))
            chunk_index += 1
        else:
            logger.debug(
                "Article too large (%d chars), requesting LLM split: %s",
                len(seg_text), article_label,
            )
            sub_texts = await _semantic_split_with_llm(seg_text)

            for sub_text in sub_texts:
                sub_text = sub_text.strip()
                if not sub_text:
                    continue
                chunks.append(_build_chunk(
                    chunk_id      = f"{source_stem}::{chunk_index:04d}",
                    chunk_text    = sub_text,
                    document_name = document_name,
                    law_number    = law_number,
                    article       = article_label,
                    section       = current_section,
                ))
                chunk_index += 1

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# LLM Semantic Split (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

async def _semantic_split_with_llm(article_text: str) -> list[str]:
    """
    Use GPT-4o-mini to split an oversized article into semantic sub-chunks.

    Falls back to paragraph splitting if LLM response is invalid.
    """
    messages = [
        SystemMessage(content=_SPLIT_SYSTEM),
        HumanMessage(content=_SPLIT_HUMAN.format(article_text=article_text)),
    ]

    try:
        response = await llm_fast.ainvoke(messages)
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed   = json.loads(raw)
        segments = parsed.get("segments", [])

        if isinstance(segments, list) and all(
            isinstance(s, str) and s.strip() for s in segments
        ):
            logger.debug("LLM produced %d semantic segments", len(segments))
            return segments

        logger.warning("LLM segments invalid — falling back to paragraph split")

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("LLM split failed (%s) — falling back to paragraph split", exc)

    return _paragraph_split_fallback(article_text)


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_by_articles(text: str) -> list[str]:
    """
    Split full document text into segments using Article markers as boundaries.
    """
    parts = _ARTICLE_SPLIT_RE.split(text)

    segments: list[str] = []
    buffer = ""

    for part in parts:
        if _ARTICLE_SPLIT_RE.fullmatch(part):
            if buffer.strip():
                segments.append(buffer.strip())
            buffer = part
        else:
            buffer += part

    if buffer.strip():
        segments.append(buffer.strip())

    if not segments:
        segments = [text.strip()]

    return segments


def _paragraph_split_fallback(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Paragraph-level fallback when LLM split fails.

    Merges short paragraphs until max_chars, with OVERLAP_CHARS context.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current.strip())
            current = current[-OVERLAP_CHARS:] + "\n\n" + para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def extract_article_number(text: str) -> str:
    """
    Extract article label from the start of a segment for legal citation.

    Matches: Article (1) | Article 1 | Article One
    Searches only the first 80 chars.
    """
    match = _ARTICLE_LABEL_RE.search(text[:80])
    if match:
        return match.group(0).strip()
    return ""


def _build_chunk(
    chunk_id:      str,
    chunk_text:    str,
    document_name: str,
    law_number:    str,
    article:       str,
    section:       str,
) -> dict:
    """Assemble a Chunk dict from its components."""
    return {
        "chunk_id":      chunk_id,
        "document_name": document_name,
        "law_number":    law_number,
        "article":       article,
        "section":       section,
        "chunk_text":    chunk_text,
        "char_count":    len(chunk_text),
    }


