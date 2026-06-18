"""
Reranker — LLM-Based Chunk Reranking for English Tax Retrieval.

Why LLM reranking (not cross-encoder):
    - Cosine similarity measures linguistic similarity, not legal relevance
    - A chunk mentioning "tax" might rank high but not answer the question
    - GPT-4o-mini understands English legal context and judges true relevance

Strategy:
    1. Receive up to 15 deduplicated chunks from retriever
    2. Ask GPT-4o-mini to score each chunk 0-10 for relevance to query
    3. Return top_n (default: 3) highest-scored chunks
    4. On any failure → fall back to original cosine similarity order

LLM used: llm_fast (GPT-4o-mini)
Response:  JSON  {"scores": [8, 3, 9, 2, 7, ...]}
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agents.shared.llm_client import llm_fast

logger = logging.getLogger(__name__)

_CHUNK_PREVIEW_CHARS = 350

_RERANK_SYSTEM = """You are a legal expert specializing in Egyptian tax law.
Your task: score each legal passage for relevance to the user's question.
Scoring guide:
- 10: directly and completely answers the question
-  7: strongly relevant, contains useful information
-  4: partially or indirectly related
-  1: not relevant to the question
Return JSON only, no extra text: {{"scores": [number, number, ...]}}
The number of scores must exactly match the number of passages."""

_RERANK_HUMAN = """Question: {query}

Legal passages:
{chunks_text}

Score each passage from 0 to 10."""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = 3,
) -> list[dict]:
    """
    Rerank retrieved chunks by legal relevance to the query.

    Skips LLM call if chunks <= top_n.
    Falls back to cosine similarity order if LLM call fails.

    Args:
        query:  Original user query.
        chunks: Deduplicated RetrievedChunk dicts from retriever.
        top_n:  Number of top chunks to return (default: 3).

    Returns:
        List of up to top_n chunks sorted by rerank score desc.
        Each chunk gets a "rerank_score" key added.
    """
    if not chunks:
        return []

    if len(chunks) <= top_n:
        for chunk in chunks:
            chunk["rerank_score"] = chunk.get("similarity", 0.0)
        return chunks

    prompt_text = build_rerank_prompt(query, chunks)

    messages = [
        SystemMessage(content=_RERANK_SYSTEM),
        HumanMessage(content=prompt_text),
    ]

    try:
        response = await llm_fast.ainvoke(messages)
        scores   = parse_rerank_scores(response.content, expected_count=len(chunks))

    except Exception as exc:
        logger.warning("Reranker LLM call failed: %s — using similarity order", exc)
        scores = [chunk.get("similarity", 0.0) for chunk in chunks]

    ranked = sorted(
        zip(scores, chunks),
        key=lambda pair: pair[0],
        reverse=True,
    )

    result = []
    for score, chunk in ranked[:top_n]:
        chunk["rerank_score"] = round(score, 4)
        result.append(chunk)

    logger.debug(
        "Reranker: %d → %d chunks | top score=%.1f",
        len(chunks), len(result), result[0]["rerank_score"] if result else 0,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_rerank_prompt(query: str, chunks: list[dict]) -> str:
    """Build the scoring prompt sent to GPT-4o-mini."""
    lines: list[str] = []

    for i, chunk in enumerate(chunks):
        article = chunk.get("article", "")
        law     = chunk.get("law_number", "")
        text    = chunk.get("chunk_text", "")[:_CHUNK_PREVIEW_CHARS]

        header = f"[{i}]"
        if article:
            header += f" {article}"
        if law:
            header += f" — {law}"

        lines.append(f"{header}\n{text}")

    chunks_text = "\n\n---\n\n".join(lines)
    return _RERANK_HUMAN.format(query=query, chunks_text=chunks_text)


def parse_rerank_scores(response_text: str, expected_count: int) -> list[float]:
    """
    Parse JSON scores array from GPT-4o-mini response.

    Handles markdown fences, extra text, wrong length.
    Falls back to uniform 5.0 scores if parsing fails entirely.
    """
    fallback = [5.0] * expected_count

    try:
        text = response_text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            logger.warning("parse_rerank_scores: no JSON object found")
            return fallback

        parsed     = json.loads(json_match.group())
        raw_scores = parsed.get("scores", [])

        if not isinstance(raw_scores, list):
            return fallback

        scores: list[float] = []
        for val in raw_scores:
            try:
                scores.append(max(0.0, min(10.0, float(val))))
            except (TypeError, ValueError):
                scores.append(5.0)

        if len(scores) < expected_count:
            scores.extend([5.0] * (expected_count - len(scores)))
        else:
            scores = scores[:expected_count]

        return scores

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("parse_rerank_scores failed: %s — using fallback", exc)
        return fallback
