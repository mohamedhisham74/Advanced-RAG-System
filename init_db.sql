-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main chunks table
CREATE TABLE IF NOT EXISTS tax_chunks (
    id            SERIAL PRIMARY KEY,
    chunk_id      TEXT NOT NULL UNIQUE,
    document_name TEXT NOT NULL DEFAULT '',
    law_number    TEXT NOT NULL DEFAULT '',
    article       TEXT NOT NULL DEFAULT '',
    section       TEXT NOT NULL DEFAULT '',
    chunk_text    TEXT NOT NULL,
    embedding     VECTOR(1536) NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- HNSW index for fast approximate nearest-neighbor search (cosine distance)
CREATE INDEX IF NOT EXISTS tax_chunks_embedding_idx
    ON tax_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text search index on chunk_text
CREATE INDEX IF NOT EXISTS tax_chunks_text_idx
    ON tax_chunks
    USING gin (to_tsvector('english', chunk_text));
