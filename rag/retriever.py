from dataclasses import dataclass

from google import genai

from rag.config import RAGConfig
from rag.embeddings import embed_query
from rag.pinecone_store import get_pinecone_index, query_chunks


@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float


def retrieve_context(query: str, config: RAGConfig | None = None) -> tuple[str, list[RetrievedChunk]]:
    config = config or RAGConfig.from_env()
    if not config.rag_enabled:
        return "", []

    client = genai.Client(api_key=config.gemini_api_key)
    query_embedding = embed_query(client, config, query)
    index = get_pinecone_index(config)
    matches = query_chunks(
        index=index,
        namespace=config.pinecone_namespace,
        query_embedding=query_embedding,
        top_k=config.top_k,
    )

    chunks = [
        RetrievedChunk(
            source=match["source"],
            text=match["text"],
            score=float(match["score"] or 0.0),
        )
        for match in matches
        if match.get("text")
    ]
    if not chunks:
        return "", []

    context_lines = []
    for idx, chunk in enumerate(chunks, start=1):
        context_lines.append(
            f"[{idx}] المصدر: {chunk.source} | درجة التطابق: {chunk.score:.3f}\n{chunk.text}"
        )

    context = "\n\n".join(context_lines)
    return context, chunks
