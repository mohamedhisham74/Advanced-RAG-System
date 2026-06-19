"""
Reranker — LLM-Based Chunk Reranking.

Scores each chunk 0-10 for relevance to the query using GPT-4o-mini,
then returns the top_n highest-scored chunks.
Falls back to cosine similarity order on any failure.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_client import llm_fast

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


async def rerank_chunks(query: str, chunks: list[dict], top_n: int = 3) -> list[dict]:
    if not chunks:
        return []

    if len(chunks) <= top_n:
        for chunk in chunks:
            chunk["rerank_score"] = chunk.get("similarity", 0.0)
        return chunks

    prompt_text = _build_rerank_prompt(query, chunks)
    messages    = [
        SystemMessage(content=_RERANK_SYSTEM),
        HumanMessage(content=prompt_text),
    ]

    try:
        response = await llm_fast.ainvoke(messages)
        scores   = _parse_rerank_scores(response.content, expected_count=len(chunks))
    except Exception as exc:
        logger.warning("Reranker failed: %s — using similarity order", exc)
        scores = [chunk.get("similarity", 0.0) for chunk in chunks]

    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)

    result = []
    for score, chunk in ranked[:top_n]:
        chunk["rerank_score"] = round(score, 4)
        result.append(chunk)

    return result


def _build_rerank_prompt(query: str, chunks: list[dict]) -> str:
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


def _parse_rerank_scores(response_text: str, expected_count: int) -> list[float]:
    fallback = [5.0] * expected_count

    try:
        text = response_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            return fallback

        parsed     = json.loads(json_match.group())
        raw_scores = parsed.get("scores", [])

        scores: list[float] = []
        for val in raw_scores:
            try:
                scores.append(max(0.0, min(10.0, float(val))))
            except (TypeError, ValueError):
                scores.append(5.0)

        if len(scores) < expected_count:
            scores.extend([5.0] * (expected_count - len(scores)))

        return scores[:expected_count]

    except Exception as exc:
        logger.warning("parse_rerank_scores failed: %s", exc)
        return fallback
