"""
Retriever — cosine similarity search against pgvector.

retrieve()       → single embedding query
retrieve_multi() → multiple embeddings, deduplicated by chunk_id (best score kept)
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_RETRIEVE_SQL = text("""
    SELECT
        chunk_id,
        document_name,
        law_number,
        article,
        section,
        chunk_text,
        metadata,
        1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
    FROM tax_chunks
    WHERE 1 - (embedding <=> CAST(:embedding AS vector)) >= :threshold
    ORDER BY similarity DESC
    LIMIT :top_k
""")


async def retrieve(
    query_embedding: list[float],
    db: AsyncSession,
    top_k: int = 5,
    threshold: float = 0.70,
) -> list[dict]:
    """Return top_k chunks above the similarity threshold for a single embedding."""
    vector_str = _fmt_vector(query_embedding)
    try:
        result = await db.execute(
            _RETRIEVE_SQL,
            {"embedding": vector_str, "threshold": threshold, "top_k": top_k},
        )
        rows = result.mappings().all()
    except Exception as exc:
        logger.error("pgvector query failed: %s", exc)
        return []

    return [_row_to_chunk(row) for row in rows]


async def retrieve_multi(
    query_embeddings: list[list[float]],
    db: AsyncSession,
    top_k: int = 5,
    threshold: float = 0.70,
) -> list[dict]:
    """
    Retrieve and deduplicate chunks across multiple query embeddings.
    Runs queries sequentially (AsyncSession does not support concurrent ops).
    """
    if not query_embeddings:
        return []

    all_chunks: list[dict] = []
    for i, emb in enumerate(query_embeddings):
        try:
            chunks = await retrieve(emb, db, top_k, threshold)
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.warning("retrieve_multi: embedding[%d] failed: %s", i, exc)

    deduped = _deduplicate(all_chunks)
    logger.debug(
        "retrieve_multi: %d embeddings → %d raw → %d deduped",
        len(query_embeddings), len(all_chunks), len(deduped),
    )
    return deduped


def _deduplicate(chunks: list[dict]) -> list[dict]:
    """Keep the highest similarity score per chunk_id, return sorted desc."""
    best: dict[str, dict] = {}
    for chunk in chunks:
        cid   = chunk["chunk_id"]
        score = chunk.get("similarity", 0.0)
        if cid not in best or score > best[cid].get("similarity", 0.0):
            best[cid] = chunk
    return sorted(best.values(), key=lambda c: c.get("similarity", 0.0), reverse=True)


def _fmt_vector(embedding: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"


def _row_to_chunk(row) -> dict:
    import json as _json

    metadata = row["metadata"]
    if isinstance(metadata, str):
        try:
            metadata = _json.loads(metadata)
        except Exception:
            metadata = {}

    return {
        "chunk_id":      row["chunk_id"],
        "document_name": row["document_name"] or "",
        "law_number":    row["law_number"] or "",
        "article":       row["article"] or "",
        "section":       row["section"] or "",
        "chunk_text":    row["chunk_text"],
        "similarity":    round(float(row["similarity"]), 4),
        "metadata":      metadata or {},
    }
