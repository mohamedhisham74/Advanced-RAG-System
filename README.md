# Advanced RAG — Egyptian Tax Law

A production-ready Retrieval-Augmented Generation (RAG) system for querying Egyptian tax law documents. Built with FastAPI, PostgreSQL + pgvector, OpenAI, and a plain HTML/CSS/JS frontend.

## Architecture

```
User query (Arabic or English)
    │
    ▼
Query Enhancer           → translate Arabic + generate 3 semantic variations
    │
    ▼
Multi-Embedding Retrieval → pgvector cosine search per embedding (HNSW index)
    │
    ▼
LLM Reranker             → GPT-4o-mini scores each chunk 0–10 for relevance
    │
    ▼
Answer Generator         → GPT-4o-mini synthesises answer with legal citations
```

## Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| API         | FastAPI + uvicorn                   |
| LLM         | OpenAI GPT-4o-mini                  |
| Embeddings  | text-embedding-3-small (1536 dims)  |
| Vector DB   | PostgreSQL 16 + pgvector (HNSW)     |
| PDF parsing | PyMuPDF (fitz)                      |
| Frontend    | HTML / CSS / Vanilla JS             |
| Deploy      | Docker + Docker Compose             |

## Getting Started

### 1. Configure environment

```bash
cp .env.example .env
# Open .env and set:  OPENAI_API_KEY=sk-...
```

### 2. Start with Docker Compose

```bash
docker compose up --build
```

- PostgreSQL + pgvector starts on port `5432` — schema applied automatically from `init_db.sql`
- FastAPI app starts on port `8000`

### 3. Open the app

| URL                           | Description     |
|-------------------------------|-----------------|
| `http://localhost:8000`       | Chat UI         |
| `http://localhost:8000/docs`  | Swagger API docs|
| `http://localhost:8000/redoc` | ReDoc API docs  |

### 4. Upload a PDF → ask questions

1. Drag a PDF onto the upload zone (sidebar → "Upload Document")
2. Click **Upload & Index** — PDF is chunked, embedded, and stored in pgvector
3. Type a question in Arabic or English and press **Enter**

## API Reference

| Method | Path          | Description                   |
|--------|---------------|-------------------------------|
| GET    | `/api/health` | Liveness probe                |
| GET    | `/api/stats`  | Indexed chunk count           |
| POST   | `/api/ingest` | Upload and index a PDF file   |
| POST   | `/api/chat`   | Ask a question (RAG pipeline) |

### POST `/api/chat`

```json
// Request
{
  "query":        "What are the income tax rates?",
  "top_k":        5,
  "threshold":    0.70,
  "rerank_top_n": 3
}

// Response
{
  "query":               "...",
  "answer":              "According to Article 8 of Law No. 91...",
  "sources":             [{ "article": "...", "law_number": "...", "excerpt": "..." }],
  "embeddings_used":     4,
  "chunks_retrieved":    12,
  "chunks_after_rerank": 3
}
```

## Project Structure

```
app/
  core/
    config.py         Settings (pydantic-settings, from .env)
    database.py       Async SQLAlchemy engine + session factory
    llm_client.py     Shared OpenAI LLM + embedding clients
  rag/
    pdf_loader.py     PyMuPDF extraction + metadata parsing
    chunker.py        3-phase hierarchical semantic chunking
    embedder.py       Batch embedding + pgvector upsert
    query_enhancer.py Arabic detection, translation, multi-query expansion
    retriever.py      Cosine similarity search with deduplication
    reranker.py       LLM relevance scoring with fallback
    generator.py      Answer synthesis with legal citations
  api/
    deps.py           FastAPI DB dependency
    routes/
      health.py       GET /api/health, GET /api/stats
      ingest.py       POST /api/ingest
      chat.py         POST /api/chat
  main.py             FastAPI app, middleware, routes, static mount

static/
  index.html          Single-page chat interface
  css/style.css       Dark theme UI
  js/app.js           Frontend: upload, chat, sources accordion

data/
  tax_knowledge_base/
    raw/              Drop PDFs here for manual ingestion
    processed/        Cleaned text (optional persistence)

init_db.sql           PostgreSQL schema + HNSW vector index
docker-compose.yml    API + database services
Dockerfile            Python 3.11-slim image
requirements.txt      Python dependencies
.env.example          Environment variable template
```

## Notes

- Re-running ingestion on the same PDF is safe — chunks upserted by `chunk_id`.
- Arabic queries are auto-detected and translated before retrieval.
- The HNSW index gives sub-millisecond approximate nearest-neighbor search.
- Uploaded PDFs are volume-mounted at `./data` and persist across restarts.
