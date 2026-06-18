"""
Embedder — Chunk Embedding Generation and pgvector Storage.

Embedding model : text-embedding-3-small (1536 dims, OpenAI)
Storage table   : tax_chunks  (pgvector VECTOR(1536) column)
Client          : agents.shared.llm_client.embeddings  — shared, do NOT re-instantiate

Two-stage design:
    1. embed_chunks()  — pure embedding, no DB touch
       Calls OpenAI in batches of BATCH_SIZE=100 (well below the 2048 API limit).
       Returns same chunk list with "embedding": list[float] added to each dict.

    2. store_chunks()  — pure DB write, no OpenAI call
       Uses raw SQL INSERT … ON CONFLICT (chunk_id) DO UPDATE SET …
       so re-running the ingestion script never creates duplicates.
       The vector is passed as a formatted string "[f1,f2,…]" and cast
       by PostgreSQL → compatible with any asyncpg / SQLAlchemy version.

Keeping the two stages separate lets tests mock either independently.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from agents.shared.llm_client import embeddings as _embeddings

logger = logging.getLogger(__name__)

BATCH_SIZE = 50    # chunks per OpenAI embed call (300k token limit per request)

# ── SQL ────────────────────────────────────────────────────────────────────────
_UPSERT_SQL = text("""
    INSERT INTO tax_chunks
        (chunk_id, document_name, law_number, article, section,
         chunk_text, embedding, metadata)
    VALUES
        (:chunk_id, :document_name, :law_number, :article, :section,
         :chunk_text, CAST(:embedding AS vector), :metadata)
    ON CONFLICT (chunk_id) DO UPDATE SET
        document_name = EXCLUDED.document_name,
        law_number    = EXCLUDED.law_number,
        article       = EXCLUDED.article,
        section       = EXCLUDED.section,
        chunk_text    = EXCLUDED.chunk_text,
        embedding     = EXCLUDED.embedding,
        metadata      = EXCLUDED.metadata
""")

_DELETE_ALL_SQL = text("DELETE FROM tax_chunks")
_COUNT_SQL      = text("SELECT COUNT(*) FROM tax_chunks")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for all chunks using text-embedding-3-small.

    Processes in batches of BATCH_SIZE to stay within OpenAI rate limits.
    Each chunk dict gets an "embedding" key added (list[float], len=1536).

    Args:
        chunks: List of Chunk dicts from chunker.chunk_all_documents().
                Must have "chunk_text" key.

    Returns:
        New list of chunk dicts — each has "embedding": list[float] added.
        Original dicts are NOT mutated.

    Raises:
        ValueError: If any chunk is missing "chunk_text".
        Exception:  Re-raises OpenAI API errors after logging.
    """
    if not chunks:
        return []

    # Validate before hitting the API
    for i, chunk in enumerate(chunks):
        if not chunk.get("chunk_text", "").strip():
            raise ValueError(f"chunk[{i}] has empty chunk_text (chunk_id={chunk.get('chunk_id')})")

    texts          = [c["chunk_text"] for c in chunks]
    total_batches  = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    embedded_chunks: list[dict] = []

    logger.info("Embedding %d chunks in %d batches (model: text-embedding-3-small)",
                len(chunks), total_batches)

    for batch_num, start in enumerate(range(0, len(texts), BATCH_SIZE), start=1):
        batch_texts = texts[start : start + BATCH_SIZE]

        logger.debug("Embedding batch %d / %d (%d texts)",
                     batch_num, total_batches, len(batch_texts))

        try:
            batch_embeddings: list[list[float]] = await _embeddings.aembed_documents(batch_texts)
        except Exception as exc:
            logger.error("OpenAI embedding failed at batch %d: %s", batch_num, exc)
            raise

        for chunk, embedding in zip(chunks[start : start + BATCH_SIZE], batch_embeddings):
            embedded_chunks.append({**chunk, "embedding": embedding})

        logger.info("  ✓ Batch %d / %d embedded", batch_num, total_batches)

    logger.info("Embedding complete — %d chunks ready for storage", len(embedded_chunks))
    return embedded_chunks


async def store_chunks(chunks: list[dict], db_session: AsyncSession) -> int:
    """
    Upsert embedded chunks into the tax_chunks pgvector table.

    Uses INSERT … ON CONFLICT (chunk_id) DO UPDATE so the ingestion script
    can be re-run safely without creating duplicates.

    The embedding (list[float]) is serialised to the pgvector string format
    "[f1,f2,…]" and cast by PostgreSQL. This works with any asyncpg version
    without requiring codec registration.

    Args:
        chunks:     Chunk dicts — must have "embedding": list[float] (len=1536).
        db_session: Async SQLAlchemy read-write session.

    Returns:
        Number of rows upserted.

    Raises:
        ValueError: If a chunk is missing its embedding.
    """
    if not chunks:
        return 0

    stored = 0

    for chunk in chunks:
        if "embedding" not in chunk:
            raise ValueError(
                f"chunk {chunk.get('chunk_id')} is missing 'embedding'. "
                "Call embed_chunks() first."
            )

        # pgvector string format: "[0.12345,0.67890,…]"
        vector_str = "[" + ",".join(f"{v:.8f}" for v in chunk["embedding"]) + "]"

        metadata = {
            "source_file": chunk.get("source_file", ""),
            "char_count":  chunk.get("char_count", len(chunk.get("chunk_text", ""))),
        }

        await db_session.execute(_UPSERT_SQL, {
            "chunk_id":      chunk["chunk_id"],
            "document_name": chunk.get("document_name", ""),
            "law_number":    chunk.get("law_number", ""),
            "article":       chunk.get("article", ""),
            "section":       chunk.get("section", ""),
            "chunk_text":    chunk["chunk_text"],
            "embedding":     vector_str,
            "metadata":      json.dumps(metadata, ensure_ascii=False),
        })
        stored += 1

    await db_session.commit()
    logger.info("Stored %d chunks in tax_chunks (upsert)", stored)
    return stored


async def clear_all_chunks(db_session: AsyncSession) -> int:
    """
    Delete all rows from the tax_chunks table.

    Called by the ingestion script with --clear flag before re-ingestion.

    Args:
        db_session: Async SQLAlchemy read-write session.

    Returns:
        Number of rows deleted.
    """
    result = await db_session.execute(_DELETE_ALL_SQL)
    await db_session.commit()
    deleted = result.rowcount
    logger.info("Cleared %d rows from tax_chunks", deleted)
    return deleted


async def count_chunks(db_session: AsyncSession) -> int:
    """
    Return the current number of chunks in the tax_chunks table.

    Args:
        db_session: Async SQLAlchemy session (read-only is sufficient).

    Returns:
        Row count as integer.
    """
    result = await db_session.execute(_COUNT_SQL)
    return result.scalar_one()
