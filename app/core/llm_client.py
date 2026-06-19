"""
Shared LLM and embedding clients — created once at import time.
All RAG modules import from here; never re-instantiate in other files.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import settings

# GPT-4o-mini: cheap + fast — used for chunking, query expansion, reranking, generation
llm_fast = ChatOpenAI(
    model=settings.llm_model,
    temperature=0,
    api_key=settings.openai_api_key,
)

# text-embedding-3-small: 1536 dims, matches pgvector column
embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    api_key=settings.openai_api_key,
)
