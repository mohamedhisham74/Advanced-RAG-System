"""
Chunker — Hierarchical Semantic Chunking for English Legal Documents.

Strategy (3 phases per document):
    Phase 1 — Structural Parsing: regex detects Article boundaries.
    Phase 2 — Size Gate: small articles kept as-is.
    Phase 3 — LLM Semantic Split: oversized articles split by GPT-4o-mini.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_client import llm_fast

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 2000
OVERLAP_CHARS   = 150

_ARTICLE_SPLIT_RE = re.compile(
    r"(\n\s*Article\s+(?:\(\s*\d+\s*\)|\d+(?:\s+(?:bis|ter))?|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))",
    re.IGNORECASE,
)

_CHAPTER_RE = re.compile(
    r"^\s*(Chapter\s+(?:[A-Z][a-z]+|\d+)(?:\s+\w+){0,4})",
    re.MULTILINE | re.IGNORECASE,
)

_SECTION_RE = re.compile(
    r"^\s*(Section\s+(?:[A-Z][a-z]+|\d+)(?:\s+\w+){0,4})",
    re.MULTILINE | re.IGNORECASE,
)

_PART_RE = re.compile(
    r"^\s*(Part\s+(?:[A-Z][a-z]+|\d+)(?:\s+\w+){0,4})",
    re.MULTILINE | re.IGNORECASE,
)

_ARTICLE_LABEL_RE = re.compile(
    r"Article\s+(?:\(\s*(\d+)\s*\)|(\d+)|([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))",
    re.IGNORECASE,
)

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


async def chunk_all_documents(documents: list[dict]) -> list[dict]:
    all_chunks: list[dict] = []
    for document in documents:
        doc_chunks = await chunk_document(document)
        all_chunks.extend(doc_chunks)
        logger.info("Chunked %s → %d chunks", document.get("document_name", "?"), len(doc_chunks))
    return all_chunks


async def chunk_document(document: dict) -> list[dict]:
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

        parsed   = json.loads(raw)
        segments = parsed.get("segments", [])

        if isinstance(segments, list) and all(isinstance(s, str) and s.strip() for s in segments):
            return segments

    except Exception as exc:
        logger.warning("LLM split failed (%s) — falling back to paragraph split", exc)

    return _paragraph_split_fallback(article_text)


def _split_by_articles(text: str) -> list[str]:
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

    return segments or [text.strip()]


def _paragraph_split_fallback(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
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

    return chunks or [text]


def extract_article_number(text: str) -> str:
    match = _ARTICLE_LABEL_RE.search(text[:80])
    if match:
        return match.group(0).strip()
    return ""


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
