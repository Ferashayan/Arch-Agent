"""
api/routes/ingest.py
--------------------
POST /api/ingest — Trigger document ingestion into Pinecone
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from core.schemas import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents_endpoint(request: IngestRequest):
    """
    Trigger document ingestion from the /documents directory into Pinecone.
    Wraps the existing ingest.py logic.
    """
    try:
        from ingest import ingest_documents

        # ingest_documents prints to stdout — we capture the count
        # by calling the underlying functions directly
        from pathlib import Path

        from rag.chunker import chunk_text
        from rag.config import RAGConfig
        from rag.embeddings import embed_documents
        from rag.pinecone_store import (
            clear_namespace,
            get_pinecone_index,
            upsert_chunks,
        )
        from ingest import collect_documents, load_document, DOCUMENTS_DIR
        from google import genai

        config = RAGConfig.from_env()
        if not config.rag_enabled:
            raise HTTPException(
                status_code=400,
                detail="RAG is not configured. Set GEMINI_API_KEY, PINECONE_API_KEY, and PINECONE_INDEX in .env.",
            )

        documents = collect_documents(DOCUMENTS_DIR)
        if not documents:
            raise HTTPException(
                status_code=404,
                detail=f"No documents found in {DOCUMENTS_DIR}",
            )

        client = genai.Client(api_key=config.gemini_api_key)
        index = get_pinecone_index(config)

        if request.reset:
            clear_namespace(index, config.pinecone_namespace)
            logger.info("Cleared namespace: %s", config.pinecone_namespace)

        total_vectors = 0
        for document_path in documents:
            text = load_document(document_path)
            chunks = chunk_text(text, config.chunk_size, config.chunk_overlap)
            if not chunks:
                continue

            embeddings = embed_documents(client, config, chunks)
            count = upsert_chunks(
                index=index,
                namespace=config.pinecone_namespace,
                chunks=chunks,
                embeddings=embeddings,
                source=document_path.name,
            )
            total_vectors += count

        return IngestResponse(
            total_vectors=total_vectors,
            message=f"Successfully upserted {total_vectors} vectors into {config.pinecone_index}/{config.pinecone_namespace}",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))
