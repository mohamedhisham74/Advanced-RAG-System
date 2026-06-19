"""
Chat endpoint — full Advanced RAG pipeline:
    query_enhancer → retriever → reranker → generator
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.config import settings
from app.rag.generator import generate_answer
from app.rag.query_enhancer import enhance_query
from app.rag.reranker import rerank_chunks
from app.rag.retriever import retrieve_multi

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    top_k: int = settings.rag_top_k
    threshold: float = settings.rag_similarity_threshold
    rerank_top_n: int = settings.rag_rerank_top_n


class SourceItem(BaseModel):
    chunk_id: str
    document_name: str
    law_number: str
    article: str
    section: str
    similarity: float
    rerank_score: float
    excerpt: str


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceItem]
    embeddings_used: int
    chunks_retrieved: int
    chunks_after_rerank: int


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    logger.info("Chat request: %r", query[:80])

    # Step 1: Query enhancement (multi-query embeddings)
    embeddings = await enhance_query(query)
    if not embeddings:
        raise HTTPException(status_code=500, detail="Failed to embed query.")

    # Step 2: Multi-embedding retrieval
    raw_chunks = await retrieve_multi(
        query_embeddings=embeddings,
        db_session=db,
        top_k=req.top_k,
        threshold=req.threshold,
    )

    # Step 3: LLM reranking
    reranked = await rerank_chunks(query, raw_chunks, top_n=req.rerank_top_n)

    # Step 4: Answer generation
    result = await generate_answer(query, reranked)

    return ChatResponse(
        query               = query,
        answer              = result["answer"],
        sources             = result["sources"],
        embeddings_used     = len(embeddings),
        chunks_retrieved    = len(raw_chunks),
        chunks_after_rerank = len(reranked),
    )
