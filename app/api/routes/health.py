from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.rag.embedder import count_chunks

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/stats")
async def stats(db: AsyncSession = Depends(get_db)):
    total = await count_chunks(db)
    return {"indexed_chunks": total}
