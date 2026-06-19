"""
Reranker — LLM-based chunk relevance scoring.

GPT-4o-mini scores each chunk 0–10 for relevance to the query.
Returns top_n highest-scored chunks.
Falls back to cosine similarity order on any LLM failure.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_client import llm_fast

logger = logging.getLogger(__name__)

_PREVIEW_CHARS = 350

_SYSTEM = """You are a legal expert specializing in Egyptian tax law.
Score each legal passage for relevance to the user's question:
- 10: directly and completely answers the question
-  7: strongly relevant, contains useful information
-  4: partially or indirectly related
-  1: not relevant
Return JSON only: {"scores": [number, ...]}
The array length must exactly match the number of passages."""

_HUMAN = """Question: {query}

Legal passages:
{chunks_text}

Score each passage 0–10."""


async def rerank_chunks(query: str, chunks: list[dict], top_n: int = 3) -> list[dict]:
    """
    Rerank chunks by LLM relevance score; return top_n.
    Skips the LLM call when len(chunks) <= top_n.
    """
    if not chunks:
        return []

    if len(chunks) <= top_n:
        for chunk in chunks:
            chunk.setdefault("rerank_score", chunk.get("similarity", 0.0))
        return chunks

    prompt = _build_prompt(query, chunks)
    try:
        response = await llm_fast.ainvoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
        scores = _parse_scores(response.content, expected=len(chunks))
    except Exception as exc:
        logger.warning("Reranker LLM failed (%s) — using similarity order", exc)
        scores = [c.get("similarity", 0.0) for c in chunks]

    ranked = sorted(zip(scores, chunks), key=lambda p: p[0], reverse=True)

    result = []
    for score, chunk in ranked[:top_n]:
        chunk["rerank_score"] = round(score, 4)
        result.append(chunk)

    logger.debug("Rerank: %d → %d | top score=%.1f", len(chunks), len(result), result[0]["rerank_score"] if result else 0)
    return result


def _build_prompt(query: str, chunks: list[dict]) -> str:
    lines: list[str] = []
    for i, chunk in enumerate(chunks):
        header = f"[{i}]"
        if chunk.get("article"):
            header += f" {chunk['article']}"
        if chunk.get("law_number"):
            header += f" — {chunk['law_number']}"
        lines.append(f"{header}\n{chunk.get('chunk_text', '')[:_PREVIEW_CHARS]}")
    return _HUMAN.format(query=query, chunks_text="\n\n---\n\n".join(lines))


def _parse_scores(response_text: str, expected: int) -> list[float]:
    fallback = [5.0] * expected
    try:
        text = response_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return fallback

        raw_scores = json.loads(m.group()).get("scores", [])
        scores: list[float] = []
        for val in raw_scores:
            try:
                scores.append(max(0.0, min(10.0, float(val))))
            except (TypeError, ValueError):
                scores.append(5.0)

        if len(scores) < expected:
            scores.extend([5.0] * (expected - len(scores)))
        return scores[:expected]

    except Exception as exc:
        logger.warning("_parse_scores failed: %s", exc)
        return fallback
