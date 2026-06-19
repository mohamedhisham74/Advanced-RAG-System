"""
Generator — Final Answer Generation with Legal Citations.

Takes the user query + top reranked chunks and produces a structured
answer with source citations using GPT-4o-mini.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_client import llm_fast

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM = """You are an expert legal advisor specializing in Egyptian tax law.
Your role: answer the user's tax question using ONLY the provided legal passages.

Rules:
- Base your answer strictly on the provided passages — do not add external knowledge
- Cite sources using [Article X — Law Y] format after each relevant point
- If the passages don't contain enough information, say so clearly
- Write in the same language as the user's question (Arabic or English)
- Be precise and professional; avoid vague or general statements
- Structure the answer with clear paragraphs"""

_ANSWER_HUMAN = """User question: {query}

Legal passages to use:
{context}

Please provide a precise answer with citations."""


async def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Generate a final answer from reranked chunks.

    Returns:
        {
            "answer":   str,         # the generated answer text
            "sources":  list[dict],  # citation metadata for each chunk used
        }
    """
    if not chunks:
        return {
            "answer":  "لم يتم العثور على معلومات كافية للإجابة على هذا السؤال في قاعدة البيانات.\n"
                       "No sufficient information found in the knowledge base to answer this question.",
            "sources": [],
        }

    context = _build_context(chunks)
    messages = [
        SystemMessage(content=_ANSWER_SYSTEM),
        HumanMessage(content=_ANSWER_HUMAN.format(query=query, context=context)),
    ]

    try:
        response = await llm_fast.ainvoke(messages)
        answer   = response.content.strip()
    except Exception as exc:
        logger.error("Answer generation failed: %s", exc)
        answer = "An error occurred while generating the answer. Please try again."

    sources = [_chunk_to_source(chunk) for chunk in chunks]

    return {"answer": answer, "sources": sources}


def _build_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        article = chunk.get("article", "")
        law     = chunk.get("law_number", "")
        doc     = chunk.get("document_name", "")
        text    = chunk.get("chunk_text", "")

        header = f"[Passage {i + 1}]"
        if article:
            header += f" {article}"
        if law:
            header += f" — {law}"
        if doc:
            header += f" ({doc})"

        parts.append(f"{header}\n{text}")

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
