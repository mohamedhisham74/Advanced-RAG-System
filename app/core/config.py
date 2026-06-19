"""
Application settings loaded from environment variables.
All secrets come from .env — never hardcode credentials here.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # ── PostgreSQL + pgvector ─────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:postgres@db:5432/ragdb"

    # ── RAG pipeline defaults ─────────────────────────────────────────────────
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.70
    rag_rerank_top_n: int = 3

    # ── Server ────────────────────────────────────────────────────────────────
    app_title: str = "Advanced RAG — Egyptian Tax Law"
    debug: bool = False


settings = Settings()
