"""
Query Enhancer — multi-query expansion for improved retrieval coverage.

Steps:
    1. Detect Arabic → translate to English via LLM
    2. Embed the (translated) query
    3. Generate 3 semantic variations via LLM and embed them
    4. Return all embeddings: [query_emb, var1_emb, var2_emb, var3_emb]
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_client import embeddings as _embeddings
from app.core.llm_client import llm_fast

logger = logging.getLogger(__name__)

_TRANSLATE_SYSTEM = """You are a professional legal translator specializing in Egyptian tax law.
Translate the following Arabic tax question into English.
Return only the English translation — no explanation, no preamble."""

_TRANSLATE_HUMAN = "Arabic question: {query}"

_VARIATIONS_SYSTEM = """You are a legal language specialist in Egyptian tax law.
Rephrase the given question in three different ways:
- Variation 1: formal legal English with statutory terminology
- Variation 2: alternative synonyms for the main concept
- Variation 3: shorter, more direct phrasing
Return JSON only: {"variations": ["...", "...", "..."]}"""

_VARIATIONS_HUMAN = "Original question: {query}"


def _is_arabic(text: str) -> bool:
    return any(unicodedata.name(ch, "").startswith("ARABIC") for ch in text if ch.strip())


async def _translate_to_english(query: str) -> str:
    try:
        response = await llm_fast.ainvoke([
            SystemMessage(content=_TRANSLATE_SYSTEM),
            HumanMessage(content=_TRANSLATE_HUMAN.format(query=query)),
        ])
        return response.content.strip()
    except Exception as exc:
        logger.warning("Translation failed (%s) — using original", exc)
        return query


async def enhance_query(query: str) -> list[list[float]]:
    """
    Return 1–4 embedding vectors for the query (raw + up to 3 variations).
    Falls back to [raw_embedding] only if variation generation fails.
    """
    raw = query.strip()

    if _is_arabic(raw):
        base_query = await _translate_to_english(raw)
        logger.info("Arabic → English: %r", base_query)
    else:
        base_query = raw

    variations_task = asyncio.create_task(_generate_variations(base_query))

    try:
        query_emb = await _embeddings.aembed_query(base_query)
    except Exception as exc:
        logger.error("Query embedding failed: %s", exc)
        return []

    try:
        variations = await variations_task
    except Exception as exc:
        logger.warning("Variations task failed: %s", exc)
        variations = []

    var_embeddings: list[list[float]] = []
    valid = [v for v in variations if v.strip()]
    if valid:
        try:
            var_embeddings = await _embeddings.aembed_documents(valid)
        except Exception as exc:
            logger.warning("Variation embedding failed: %s", exc)

    all_embeddings = [query_emb] + var_embeddings
    logger.debug("enhance_query → %d embeddings (1 raw + %d variations)", len(all_embeddings), len(var_embeddings))
    return all_embeddings


async def _generate_variations(query: str) -> list[str]:
    try:
        response = await llm_fast.ainvoke([
            SystemMessage(content=_VARIATIONS_SYSTEM),
            HumanMessage(content=_VARIATIONS_HUMAN.format(query=query)),
        ])
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        variations = json.loads(raw).get("variations", [])
        valid = [v.strip() for v in variations if isinstance(v, str) and v.strip()]
        return valid[:3] if len(valid) >= 2 else []

    except Exception as exc:
        logger.warning("_generate_variations failed: %s", exc)
        return []
