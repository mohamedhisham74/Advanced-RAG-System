"""
Embedder — chunk embedding generation and pgvector upsert.

Embedding model : text-embedding-3-small (1536 dims)
Storage table   : tax_chunks — VECTOR(1536) column
Upsert strategy : ON CONFLICT (chunk_id) DO UPDATE — safe to re-run ingestion.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm_client import embeddings as _embeddings

logger = logging.getLogger(__name__)

BATCH_SIZE = 50   # well under the 2048-input OpenAI limit

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


async def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add "embedding": list[float] to every chunk dict.
    Processes in batches of BATCH_SIZE. Returns new list; originals not mutated.
    """
    if not chunks:
        return []

    for i, chunk in enumerate(chunks):
        if not chunk.get("chunk_text", "").strip():
            raise ValueError(f"chunk[{i}] has empty chunk_text (id={chunk.get('chunk_id')})")

    texts         = [c["chunk_text"] for c in chunks]
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    embedded: list[dict] = []

    logger.info("Embedding %d chunks in %d batch(es)", len(chunks), total_batches)

    for batch_num, start in enumerate(range(0, len(texts), BATCH_SIZE), start=1):
        batch_texts = texts[start : start + BATCH_SIZE]
        try:
            vectors = await _embeddings.aembed_documents(batch_texts)
        except Exception as exc:
            logger.error("Embedding batch %d failed: %s", batch_num, exc)
            raise

        for chunk, vec in zip(chunks[start : start + BATCH_SIZE], vectors):
            embedded.append({**chunk, "embedding": vec})

        logger.info("  Batch %d / %d done", batch_num, total_batches)

    return embedded


async def store_chunks(chunks: list[dict], db: AsyncSession) -> int:
    """Upsert embedded chunks into tax_chunks. Returns number of rows stored."""
    if not chunks:
        return 0

    stored = 0
    for chunk in chunks:
        if "embedding" not in chunk:
            raise ValueError(f"chunk {chunk.get('chunk_id')} missing 'embedding' — call embed_chunks() first")

        vector_str = "[" + ",".join(f"{v:.8f}" for v in chunk["embedding"]) + "]"
        metadata   = {
            "source_file": chunk.get("source_file", ""),
            "char_count":  chunk.get("char_count", len(chunk.get("chunk_text", ""))),
        }

        await db.execute(_UPSERT_SQL, {
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

    await db.commit()
    logger.info("Stored %d chunks (upsert)", stored)
    return stored


async def clear_all_chunks(db: AsyncSession) -> int:
    """Delete all rows from tax_chunks. Returns deleted row count."""
    result = await db.execute(_DELETE_ALL_SQL)
    await db.commit()
    return result.rowcount


async def count_chunks(db: AsyncSession) -> int:
    """Return current row count in tax_chunks."""
    result = await db.execute(_COUNT_SQL)
    return result.scalar_one()
