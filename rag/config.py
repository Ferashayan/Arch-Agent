import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class RAGConfig:
    gemini_api_key: str
    gemini_model: str
    embedding_model: str
    embedding_dimension: int
    pinecone_api_key: str
    pinecone_index: str
    pinecone_namespace: str
    top_k: int
    chunk_size: int
    chunk_overlap: int

    @property
    def rag_enabled(self) -> bool:
        return bool(self.gemini_api_key and self.pinecone_api_key and self.pinecone_index)

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip("'\""),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip("'\""),
            embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004").strip("'\""),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1024")),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", "").strip("'\""),
            pinecone_index=os.getenv("PINECONE_INDEX", "arch-agent").strip("'\""),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "saudi-building-code").strip("'\""),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "120")),
        )

