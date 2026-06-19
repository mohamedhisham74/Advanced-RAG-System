"""
Generator — final answer generation with legal citations.

Accepts the user query + top reranked chunks.
Produces a natural-language answer with [Article X — Law Y] citations.
Responds in the same language as the query (Arabic or English).
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_client import llm_fast

logger = logging.getLogger(__name__)

_SYSTEM = """You are an expert legal advisor specializing in Egyptian tax law.
Answer the user's question using ONLY the provided legal passages below.

Rules:
- Base your answer strictly on the provided passages — no external knowledge
- After each relevant point, cite the source: [Article X — Law Y]
- If passages don't contain enough information, say so clearly
- Respond in the same language as the user's question (Arabic or English)
- Write in clear, professional paragraphs"""

_HUMAN = """User question: {query}

Legal passages:
{context}

Provide a precise answer with citations."""

_NO_ANSWER_MSG = (
    "لم يتم العثور على معلومات كافية في قاعدة البيانات للإجابة على هذا السؤال.\n\n"
    "No sufficient information found in the knowledge base to answer this question. "
    "Please upload relevant PDF documents first."
)


async def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Generate a final answer from reranked chunks.

    Returns:
        { "answer": str, "sources": list[dict] }
    """
    if not chunks:
        return {"answer": _NO_ANSWER_MSG, "sources": []}

    context = _build_context(chunks)
    try:
        response = await llm_fast.ainvoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=_HUMAN.format(query=query, context=context)),
        ])
        answer = response.content.strip()
    except Exception as exc:
        logger.error("Answer generation failed: %s", exc)
        answer = "An error occurred while generating the answer. Please try again."

    return {
        "answer":  answer,
        "sources": [_chunk_to_source(c) for c in chunks],
    }


def _build_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        parts_header = [f"[Passage {i + 1}]"]
        if chunk.get("article"):
            parts_header.append(chunk["article"])
        if chunk.get("law_number"):
            parts_header.append(f"— {chunk['law_number']}")
        if chunk.get("document_name"):
            parts_header.append(f"({chunk['document_name']})")

        parts.append(" ".join(parts_header) + "\n" + chunk.get("chunk_text", ""))

    return "\n\n---\n\n".join(parts)


def _chunk_to_source(chunk: dict) -> dict:
    return {
        "chunk_id":      chunk.get("chunk_id", ""),
        "document_name": chunk.get("document_name", ""),
        "law_number":    chunk.get("law_number", ""),
        "article":       chunk.get("article", ""),
        "section":       chunk.get("section", ""),
        "similarity":    chunk.get("similarity", 0.0),
        "rerank_score":  chunk.get("rerank_score", 0.0),
        "excerpt":       chunk.get("chunk_text", "")[:300],
    }
