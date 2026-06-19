"""
Chunker — Hierarchical Semantic Chunking for English Legal Documents.

Phase 1: Structural split at Article boundaries (regex, free).
Phase 2: Size gate — articles under MAX_CHUNK_CHARS kept as-is.
Phase 3: Oversized articles split semantically by GPT-4o-mini (JSON output).
         Falls back to paragraph splitting if LLM response is malformed.

Output Chunk dict keys:
    chunk_id, document_name, law_number, article, section, chunk_text, char_count
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.core.llm_client import llm_fast

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = settings.rag_max_chunk_chars
OVERLAP_CHARS   = 150

_ARTICLE_SPLIT_RE = re.compile(
    r"(\n\s*Article\s+(?:\(\s*\d+\s*\)|\d+(?:\s+(?:bis|ter))?|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))",
    re.IGNORECASE,
)
_CHAPTER_RE = re.compile(r"^\s*(Chapter\s+(?:[A-Z][a-z]+|\d+)(?:\s+\w+){0,4})", re.MULTILINE | re.IGNORECASE)
_SECTION_RE = re.compile(r"^\s*(Section\s+(?:[A-Z][a-z]+|\d+)(?:\s+\w+){0,4})", re.MULTILINE | re.IGNORECASE)
_PART_RE    = re.compile(r"^\s*(Part\s+(?:[A-Z][a-z]+|\d+)(?:\s+\w+){0,4})", re.MULTILINE | re.IGNORECASE)
_ARTICLE_LABEL_RE = re.compile(
    r"Article\s+(?:\(\s*(\d+)\s*\)|(\d+)|([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))",
    re.IGNORECASE,
)

_SPLIT_SYSTEM = """You are an expert legal document analyzer specializing in tax law.
Split the provided legal article text into semantically coherent segments.
Rules:
- Copy text verbatim — do NOT modify any word
- Each segment covers one complete legal concept or sub-provision
- Number of segments: 2 to 4
- Return JSON only: {"segments": ["...", "..."]}"""

_SPLIT_HUMAN = 'Split into semantic segments and return JSON.\n\nArticle text:\n{article_text}'


async def chunk_all_documents(documents: list[dict]) -> list[dict]:
    """Chunk all ProcessedDocuments and return a flat list of Chunk dicts."""
    all_chunks: list[dict] = []
    for doc in documents:
        chunks = await chunk_document(doc)
        all_chunks.extend(chunks)
        logger.info("Chunked '%s' → %d chunks", doc.get("document_name", "?"), len(chunks))
    return all_chunks


async def chunk_document(document: dict) -> list[dict]:
    """Split a single ProcessedDocument into semantic chunks."""
    text          = document.get("raw_text", "")
    document_name = document.get("document_name", "")
    law_number    = document.get("law_number", "")
    source_stem   = Path(document.get("source_file", "doc")).stem

    if not text.strip():
        return []

    article_segments = _split_by_articles(text)

    chunks: list[dict] = []
    chunk_index     = 0
    current_section = ""

    for segment in article_segments:
        seg_text = segment.strip()
        if not seg_text:
            continue

        section_match = _CHAPTER_RE.search(seg_text) or _SECTION_RE.search(seg_text) or _PART_RE.search(seg_text)
        if section_match:
            current_section = section_match.group(1).strip()

        article_label = _extract_article_label(seg_text)

        if len(seg_text) <= MAX_CHUNK_CHARS:
            chunks.append(_build_chunk(
                f"{source_stem}::{chunk_index:04d}", seg_text,
                document_name, law_number, article_label, current_section,
            ))
            chunk_index += 1
        else:
            sub_texts = await _semantic_split_with_llm(seg_text)
            for sub in sub_texts:
                sub = sub.strip()
                if not sub:
                    continue
                chunks.append(_build_chunk(
                    f"{source_stem}::{chunk_index:04d}", sub,
                    document_name, law_number, article_label, current_section,
                ))
                chunk_index += 1

    return chunks


async def _semantic_split_with_llm(article_text: str) -> list[str]:
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

        segments = json.loads(raw).get("segments", [])
        if isinstance(segments, list) and all(isinstance(s, str) and s.strip() for s in segments):
            return segments

    except Exception as exc:
        logger.warning("LLM split failed (%s) — fallback to paragraph split", exc)

    return _paragraph_split_fallback(article_text)


def _split_by_articles(text: str) -> list[str]:
    parts    = _ARTICLE_SPLIT_RE.split(text)
    segments = []
    buffer   = ""

    for part in parts:
        if _ARTICLE_SPLIT_RE.fullmatch(part):
            if buffer.strip():
                segments.append(buffer.strip())
            buffer = part
        else:
            buffer += part

    if buffer.strip():
        segments.append(buffer.strip())

    return segments or [text.strip()]


def _paragraph_split_fallback(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks  = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > MAX_CHUNK_CHARS:
            chunks.append(current.strip())
            current = current[-OVERLAP_CHARS:] + "\n\n" + para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text]


def _extract_article_label(text: str) -> str:
    m = _ARTICLE_LABEL_RE.search(text[:80])
    return m.group(0).strip() if m else ""


def _build_chunk(chunk_id, chunk_text, document_name, law_number, article, section) -> dict:
    return {
        "chunk_id":      chunk_id,
        "document_name": document_name,
        "law_number":    law_number,
        "article":       article,
        "section":       section,
        "chunk_text":    chunk_text,
        "char_count":    len(chunk_text),
    }
