"""
Retriever — Cosine Similarity Search from pgvector.

Responsibility:
    - Query tax_chunks using pgvector cosine distance operator (<=>)
    - Filter chunks below similarity threshold
    - Support multiple embeddings (HyDE + multi-query) via retrieve_multi()
    - Deduplicate by chunk_id, keeping highest similarity score

SQL (cosine similarity):
    1 - (embedding <=> :vec)  →  similarity score in [0, 1]
    Ordered DESC, filtered by threshold, limited to top_k.

Config defaults (from backend.core.config):
    rag_top_k                = 5
    rag_similarity_threshold = 0.75
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ── SQL ────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def retrieve(
    query_embedding: list[float],
    db_session: AsyncSession,
    top_k: int = 4,
    threshold: float = 0.75,
) -> list[dict]:
    """
    Retrieve top_k chunks most similar to a single query embedding.

    Args:
        query_embedding: 1536-dim float vector from embedder or query_enhancer.
        db_session:      Async SQLAlchemy session (read-only is sufficient).
        top_k:           Max number of chunks to return.
        threshold:       Min cosine similarity to include (0.0 – 1.0).

    Returns:
        List of RetrievedChunk dicts sorted by similarity descending:
        {
            "chunk_id":      str,
            "document_name": str,
            "law_number":    str,
            "article":       str,
            "section":       str,
            "chunk_text":    str,
            "similarity":    float,
            "metadata":      dict,
        }
        Returns [] if no chunks meet the threshold.
    """
    vector_str = _format_vector(query_embedding)

    try:
        result = await db_session.execute(
            _RETRIEVE_SQL,
            {
                "embedding": vector_str,
                "threshold": threshold,
                "top_k":     top_k,
            },
        )
        rows = result.mappings().all()

    except Exception as exc:
        logger.error("pgvector retrieve failed: %s", exc)
        return []

    chunks = [_row_to_chunk(row) for row in rows]

    logger.debug(
        "retrieve: top_k=%d threshold=%.2f → %d chunks returned",
        top_k, threshold, len(chunks),
    )
    return chunks


async def retrieve_multi(
    query_embeddings: list[list[float]],
    db_session: AsyncSession,
    top_k: int = 4,
    threshold: float = 0.75,
) -> list[dict]:
    """
    Retrieve and deduplicate chunks across multiple query embeddings.

    Runs all individual retrieve() calls concurrently via asyncio.gather,
    then merges and deduplicates results keeping the highest similarity
    score per chunk_id.

    Typical input: [hyde_embedding, var1_emb, var2_emb, var3_emb]

    Args:
        query_embeddings: List of 1536-dim float vectors.
        db_session:       Async SQLAlchemy session.
        top_k:            Max chunks per individual embedding query.
        threshold:        Min cosine similarity threshold.

    Returns:
        Deduplicated list of RetrievedChunk dicts sorted by similarity desc.
    """
    if not query_embeddings:
        return []

    # Run sequentially — AsyncSession does not allow concurrent operations
    # on the same connection (asyncio.gather causes pgBouncer errors in
    # transaction mode: "session is provisioning a new connection").
    all_chunks: list[dict] = []
    for i, emb in enumerate(query_embeddings):
        try:
            chunks = await retrieve(emb, db_session, top_k, threshold)
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.warning("retrieve_multi: embedding[%d] failed: %s", i, exc)

    deduplicated = deduplicate_chunks(all_chunks)

    logger.debug(
        "retrieve_multi: %d embeddings → %d raw → %d deduped chunks",
        len(query_embeddings), len(all_chunks), len(deduplicated),
    )
    return deduplicated


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """
    Merge duplicate chunk_ids keeping the highest similarity score.

    Called after retrieve_multi() combines results from several embeddings.

    Args:
        chunks: Raw list that may contain repeated chunk_ids.

    Returns:
        Unique chunks sorted by similarity descending.
    """
    best: dict[str, dict] = {}

    for chunk in chunks:
        cid   = chunk["chunk_id"]
        score = chunk.get("similarity", 0.0)

        if cid not in best or score > best[cid].get("similarity", 0.0):
            best[cid] = chunk

    return sorted(best.values(), key=lambda c: c.get("similarity", 0.0), reverse=True)


def _format_vector(embedding: list[float]) -> str:
    """
    Serialize a float list to pgvector string format: "[f1,f2,…]".

    PostgreSQL casts this string to VECTOR(1536) via CAST(:embedding AS vector).

    Args:
        embedding: 1536-dim float list.

    Returns:
        pgvector-compatible string.
    """
    return "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"


def _row_to_chunk(row) -> dict:
    """
    Convert a SQLAlchemy row mapping to a RetrievedChunk dict.

    Args:
        row: Row mapping from db_session.execute().mappings().

    Returns:
        RetrievedChunk dict with all required keys.
    """
    metadata = row["metadata"]
    if isinstance(metadata, str):
        import json
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
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
