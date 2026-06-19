from fastapi import APIRouter

from app.rag.embedder import count_chunks
from app.core.database import get_db

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/stats")
async def stats():
    """Return the number of indexed chunks."""
    async for db in get_db():
        total = await count_chunks(db)
        return {"indexed_chunks": total}
