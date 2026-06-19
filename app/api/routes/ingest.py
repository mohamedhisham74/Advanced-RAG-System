"""
POST /api/ingest — upload a PDF and run the full ingestion pipeline:
    load_pdf → chunk_all_documents → embed_chunks → store_chunks
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.rag.chunker import chunk_all_documents
from app.rag.embedder import embed_chunks, store_chunks
from app.rag.pdf_loader import load_pdf

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingest"])


class IngestResponse(BaseModel):
    filename:       str
    document_name:  str
    law_number:     str
    pages:          int
    chunks_created: int
    chunks_stored:  int


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    db:   AsyncSession = Depends(get_db),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted.")

    logger.info("Ingesting: %s", file.filename)

    # Write upload to a temp file so PyMuPDF can open it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        document = load_pdf(tmp_path)
        document["source_file"] = file.filename   # preserve original filename

        chunks = await chunk_all_documents([document])
        if not chunks:
            raise HTTPException(status_code=422, detail="No chunks produced from this PDF.")

        embedded = await embed_chunks(chunks)
        stored   = await store_chunks(embedded, db)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(
        filename       = file.filename,
        document_name  = document["document_name"],
        law_number     = document["law_number"],
        pages          = document["pages"],
        chunks_created = len(chunks),
        chunks_stored  = stored,
    )
