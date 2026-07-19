from __future__ import annotations

import hashlib
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from rag.config import RAGConfig


def get_pinecone_index(config: RAGConfig):
    pc = Pinecone(api_key=config.pinecone_api_key)
    existing = {index.name for index in pc.list_indexes()}
    if config.pinecone_index not in existing:
        pc.create_index(
            name=config.pinecone_index,
            dimension=config.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(config.pinecone_index)


def make_vector_id(source: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(f"{source}:{chunk_index}:{text}".encode("utf-8")).hexdigest()
    return digest[:32]


def upsert_chunks(
    index,
    namespace: str,
    chunks: list[str],
    embeddings: list[list[float]],
    source: str,
) -> int:
    vectors: list[tuple[str, list[float], dict[str, Any]]] = []
    for chunk_index, (text, values) in enumerate(zip(chunks, embeddings)):
        vector_id = make_vector_id(source, chunk_index, text)
        vectors.append(
            (
                vector_id,
                values,
                {
                    "source": source,
                    "chunk_index": chunk_index,
                    "text": text,
                },
            )
        )

    batch_size = 100
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size]
        index.upsert(vectors=batch, namespace=namespace)

    return len(vectors)


def query_chunks(
    index,
    namespace: str,
    query_embedding: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    response = index.query(
        namespace=namespace,
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )
    matches = []
    for match in response.matches or []:
        metadata = match.metadata or {}
        matches.append(
            {
                "id": match.id,
                "score": match.score,
                "source": metadata.get("source", "unknown"),
                "text": metadata.get("text", ""),
            }
        )
    return matches


def clear_namespace(index, namespace: str) -> None:
    index.delete(delete_all=True, namespace=namespace)
