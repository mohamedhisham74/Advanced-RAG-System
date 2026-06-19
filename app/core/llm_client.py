"""
Shared LLM and embedding clients — instantiated once at startup.
Import llm_fast and embeddings from here; never re-instantiate elsewhere.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import settings

llm_fast = ChatOpenAI(
    model=settings.llm_model,
    temperature=0,
    api_key=settings.openai_api_key,
)

embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    api_key=settings.openai_api_key,
)
