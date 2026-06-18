"""
Query Enhancer — Multi-Query Expansion for English Tax Queries.

Techniques applied:
    1. Arabic detection & translation — if the query is Arabic, translate to English first
    2. Raw query embedding  — embed the (translated) query directly
    3. Multi-Query Expansion — generate 3 query variations using LLM

LLM used: llm_fast (GPT-4o-mini) — query enhancement is low-stakes
Embedder: agents.shared.llm_client.embeddings (text-embedding-3-small)

Output of enhance_query():
    Up to 4 embeddings:
        [query_embedding, var1_embedding, var2_embedding, var3_embedding]
    Falls back to [query_embedding] alone if variations fail.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata

from langchain_core.messages import HumanMessage, SystemMessage

from agents.shared.llm_client import embeddings as _embeddings
from agents.shared.llm_client import llm_fast

logger = logging.getLogger(__name__)

# ── LLM prompts ────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_arabic(text: str) -> bool:
    """Return True if the text contains Arabic script characters."""
    return any(unicodedata.name(ch, "").startswith("ARABIC") for ch in text if ch.strip())


async def translate_to_english(query: str) -> str:
    """
    Translate an Arabic query to English using the LLM.

    Returns the English translation, or the original query on failure.
    """
    messages = [
        SystemMessage(content=_TRANSLATION_SYSTEM),
        HumanMessage(content=_TRANSLATION_HUMAN.format(query=query)),
    ]
    try:
        response = await llm_fast.ainvoke(messages)
        translated = response.content.strip()
        logger.debug("Arabic→English translation: %r → %r", query, translated)
        return translated
    except Exception as exc:
        logger.warning("Translation failed, using original query: %s", exc)
        return query


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def enhance_query(query: str, language: str = "en") -> list[list[float]]:
    """
    Query enhancement: embed raw query + up to 3 LLM-generated variations.

    If the query is Arabic, it is translated to English first before
    embedding and variation generation.

    Args:
        query:    Raw user query (Arabic or English).
        language: "en" | "ar" — overridden by automatic Arabic detection.

    Returns:
        List of 1-4 embedding vectors (list[float], len=1536 each):
            [query_embedding, var1_embedding, var2_embedding, var3_embedding]
        Falls back to [query_embedding] if variations fail.
    """
    raw = query.strip() or query

    if _is_arabic(raw):
        logger.info("Arabic query detected — translating to English before embedding.")
        base_query = await translate_to_english(raw)
    else:
        base_query = raw

    # Generate variations concurrently while embedding the raw query
    variations_task = asyncio.create_task(generate_query_variations(base_query))

    try:
        query_embedding = await _embeddings.aembed_query(base_query)
    except Exception as exc:
        logger.error("Failed to embed raw query: %s", exc)
        return []

    try:
        variations = await variations_task
    except Exception as exc:
        logger.warning("Variations task failed: %s", exc)
        variations = []

    if isinstance(variations, Exception):
        logger.warning("Multi-query failed: %s — skipping variations", variations)
        variations = []

    valid_variations = [v for v in variations if v.strip()]
    variation_embeddings: list[list[float]] = []

    if valid_variations:
        try:
            variation_embeddings = await _embeddings.aembed_documents(valid_variations)
        except Exception as exc:
            logger.warning("Variation embedding failed: %s", exc)

    result = [query_embedding] + variation_embeddings

    logger.debug(
        "enhance_query → %d embeddings (1 raw + %d variations)",
        len(result), len(variation_embeddings),
    )
    return result


async def generate_query_variations(query: str) -> list[str]:
    """
    Generate 3 English query variations using GPT-4o-mini.

    Variation strategy:
        1. Formal legal English with statutory terminology
        2. English with alternative synonyms
        3. Shorter, more direct rephrasing

    Returns:
        List of up to 3 non-empty variation strings.
        Returns [] on any error.
    """
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

        valid = [v.strip() for v in variations if isinstance(v, str) and v.strip()]

        if len(valid) < 2:
            logger.warning("Too few valid variations (%d) — skipping", len(valid))
            return []

        logger.debug("Multi-query: %d variations generated", len(valid))
        return valid[:3]

    except Exception as exc:
        logger.warning("generate_query_variations failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility shim — keep name for any callers that import it
# ─────────────────────────────────────────────────────────────────────────────

def normalize_arabic_query(query: str) -> str:
    """Pass-through — normalization is no longer needed for English queries."""
    return query.strip() if query else ""
