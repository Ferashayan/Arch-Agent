from google import genai
from google.genai import types
from pinecone import Pinecone

from rag.config import RAGConfig


def embed_documents(client: genai.Client, config: RAGConfig, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model_name = config.embedding_model.lower()
    if "llama" in model_name or "pinecone" in model_name or "e5" in model_name:
        pc = Pinecone(api_key=config.pinecone_api_key)
        res = pc.inference.embed(
            model=config.embedding_model,
            inputs=texts,
            parameters={"input_type": "passage", "truncate": "END"},
        )
        return [item.values if hasattr(item, "values") else item["values"] for item in res.data]

    result = client.models.embed_content(
        model=config.embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=config.embedding_dimension,
        ),
    )
    return [embedding.values for embedding in result.embeddings]


def embed_query(client: genai.Client, config: RAGConfig, query: str) -> list[float]:
    model_name = config.embedding_model.lower()
    if "llama" in model_name or "pinecone" in model_name or "e5" in model_name:
        pc = Pinecone(api_key=config.pinecone_api_key)
        res = pc.inference.embed(
            model=config.embedding_model,
            inputs=[query],
            parameters={"input_type": "query", "truncate": "END"},
        )
        first = res.data[0]
        return first.values if hasattr(first, "values") else first["values"]

    result = client.models.embed_content(
        model=config.embedding_model,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=config.embedding_dimension,
        ),
    )
    return result.embeddings[0].values

