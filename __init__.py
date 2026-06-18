"""
RAG Package — Advanced Retrieval-Augmented Generation for Tax Law.

Modules:
    pdf_loader      — Arabic PDF extraction + text cleaning
    chunker         — Hierarchical semantic chunking (Article-level)
    embedder        — text-embedding-3-small + pgvector storage
    retriever       — Cosine similarity search from pgvector
    query_enhancer  — HyDE + Multi-Query expansion
    reranker        — LLM-based chunk reranking

Usage:
    from backend.services.rag import pdf_loader, chunker, embedder
    from backend.services.rag import retriever, query_enhancer, reranker
"""
