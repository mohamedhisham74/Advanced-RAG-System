# Advanced RAG

Advanced RAG is a FastAPI application for ingesting PDF documents, storing semantic chunks in PostgreSQL with pgvector, and answering questions with source-backed retrieval.

## Features

- PDF upload and ingestion through a REST API
- Semantic chunking, embeddings, retrieval, reranking, and answer generation
- PostgreSQL + pgvector vector storage
- Static web interface served by FastAPI
- Docker Compose setup for the API and database

## Tech Stack

- FastAPI
- OpenAI API
- LangChain
- PostgreSQL with pgvector
- PyMuPDF
- Docker and Docker Compose

## Getting Started

### 1. Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Then update `.env` with your OpenAI API key:

```env
OPENAI_API_KEY=sk-...
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

The application will be available at:

- Web UI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health`

## API Endpoints

- `GET /api/health` - check service status
- `GET /api/stats` - return the number of indexed chunks
- `POST /api/ingest` - upload and index a PDF file
- `POST /api/chat` - ask a question against indexed documents

## Project Structure

```text
app/
  api/          FastAPI routes and dependencies
  core/         configuration, database, and LLM clients
  rag/          ingestion, retrieval, reranking, and generation pipeline
static/         frontend assets
data/           persisted uploaded data
init_db.sql     database schema and pgvector setup
```

## Notes

- Uploaded PDFs are persisted in `data/` when running with Docker Compose.
- The database is initialized from `init_db.sql`.
- Runtime secrets should be stored in `.env`, not committed to git.
