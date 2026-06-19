"""
FastAPI application entry point.

Routes:
    GET  /api/health       — liveness probe
    GET  /api/stats        — indexed chunk count
    POST /api/chat         — Advanced RAG query
    POST /api/ingest       — PDF upload & ingestion
    GET  /                 — Serve frontend (index.html)
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, health, ingest
from app.core.config import settings

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)

app = FastAPI(
    title=settings.app_title,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(ingest.router, prefix="/api", tags=["ingest"])

# Serve static frontend at /
app.mount("/", StaticFiles(directory="static", html=True), name="static")
