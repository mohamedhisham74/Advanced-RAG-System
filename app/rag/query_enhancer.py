"""
Query Enhancer — Multi-Query Expansion for Tax Queries.

Techniques:
    1. Arabic detection & translation to English
    2. Raw query embedding
    3. Multi-Query Expansion — 3 LLM-generated variations
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

_TRANSLATION_SYSTEM = """You are a professional legal translator specializing in Egyptian tax law.
Translate the following Arabic tax question into English.
Return only the English translation — no explanation, no preamble."""

_TRANSLATION_HUMAN = "Arabic question: {query}"

_VARIATIONS_SYSTEM = """You are a legal language specialist in Egyptian tax law.
Your task: rephrase a tax question in three different ways.
Rules:
- Variation 1: formal legal English using statutory terminology
- Variation 2: English with alternative synonyms for the main concept
- Variation 3: a shorter, more direct phrasing of the same question
Return JSON only: {{"variations": ["variation 1", "variation 2", "variation 3"]}}"""

_VARIATIONS_HUMAN = "Original question: {query}"


def _is_arabic(text: str) -> bool:
    return any(unicodedata.name(ch, "").startswith("ARABIC") for ch in text if ch.strip())


async def translate_to_english(query: str) -> str:
    messages = [
        SystemMessage(content=_TRANSLATION_SYSTEM),
        HumanMessage(content=_TRANSLATION_HUMAN.format(query=query)),
    ]
    try:
        response = await llm_fast.ainvoke(messages)
        return response.content.strip()
    except Exception as exc:
        logger.warning("Translation failed: %s", exc)
        return query


async def enhance_query(query: str) -> list[list[float]]:
    """
    Returns list of embeddings: [raw_query, var1, var2, var3].
    Falls back to [raw_query] only if variations fail.
    """
    raw = query.strip()

    if _is_arabic(raw):
        base_query = await translate_to_english(raw)
        logger.info("Arabic query translated: %r", base_query)
    else:
        base_query = raw

    variations_task = asyncio.create_task(generate_query_variations(base_query))

    try:
        query_embedding = await _embeddings.aembed_query(base_query)
    except Exception as exc:
        logger.error("Failed to embed query: %s", exc)
        return []

    try:
        variations = await variations_task
    except Exception as exc:
        logger.warning("Variations failed: %s", exc)
        variations = []

    valid_variations = [v for v in variations if v.strip()]
    variation_embeddings: list[list[float]] = []

    if valid_variations:
        try:
            variation_embeddings = await _embeddings.aembed_documents(valid_variations)
        except Exception as exc:
            logger.warning("Variation embedding failed: %s", exc)

    return [query_embedding] + variation_embeddings


async def generate_query_variations(query: str) -> list[str]:
    messages = [
        SystemMessage(content=_VARIATIONS_SYSTEM),
        HumanMessage(content=_VARIATIONS_HUMAN.format(query=query)),
    ]

    try:
        response = await llm_fast.ainvoke(messages)
        raw      = response.content.strip()

        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed     = json.loads(raw)
        variations = parsed.get("variations", [])
        valid      = [v.strip() for v in variations if isinstance(v, str) and v.strip()]

        return valid[:3] if len(valid) >= 2 else []

    except Exception as exc:
        logger.warning("generate_query_variations failed: %s", exc)
        return []
