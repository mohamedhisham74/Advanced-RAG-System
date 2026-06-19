"""
POST /api/chat — full Advanced RAG pipeline:
    enhance_query → retrieve_multi → rerank_chunks → generate_answer
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.config import settings
from app.rag.generator import generate_answer
from app.rag.query_enhancer import enhance_query
from app.rag.reranker import rerank_chunks
from app.rag.retriever import retrieve_multi

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    query:        str   = Field(..., min_length=1, max_length=2000)
    top_k:        int   = Field(default=settings.rag_top_k, ge=1, le=20)
    threshold:    float = Field(default=settings.rag_similarity_threshold, ge=0.0, le=1.0)
    rerank_top_n: int   = Field(default=settings.rag_rerank_top_n, ge=1, le=10)


class SourceItem(BaseModel):
    chunk_id:      str
    document_name: str
    law_number:    str
    article:       str
    section:       str
    similarity:    float
    rerank_score:  float
    excerpt:       str


class ChatResponse(BaseModel):
    query:               str
    answer:              str
    sources:             list[SourceItem]
    embeddings_used:     int
    chunks_retrieved:    int
    chunks_after_rerank: int


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    query = req.query.strip()
    logger.info("Chat: %r", query[:80])

    # 1. Query enhancement
    embeddings = await enhance_query(query)
    if not embeddings:
        raise HTTPException(status_code=500, detail="Failed to embed query.")

    # 2. Multi-embedding retrieval
    raw_chunks = await retrieve_multi(
        query_embeddings=embeddings,
        db=db,
        top_k=req.top_k,
        threshold=req.threshold,
    )

    # 3. LLM reranking
    reranked = await rerank_chunks(query, raw_chunks, top_n=req.rerank_top_n)

    # 4. Answer generation
    result = await generate_answer(query, reranked)

    return ChatResponse(
        query               = query,
        answer              = result["answer"],
        sources             = result["sources"],
        embeddings_used     = len(embeddings),
        chunks_retrieved    = len(raw_chunks),
        chunks_after_rerank = len(reranked),
    )
