"""Index local documents into Pinecone for RAG."""

from __future__ import annotations

import argparse
from pathlib import Path

from google import genai

from rag.chunker import chunk_text
from rag.config import RAGConfig
from rag.embeddings import embed_documents
from rag.pinecone_store import clear_namespace, get_pinecone_index, upsert_chunks

DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")
    if suffix == ".md":
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def collect_documents(documents_dir: Path) -> list[Path]:
    patterns = ("*.txt", "*.md", "*.pdf")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(documents_dir.glob(pattern)))
    return files


def ingest_documents(reset: bool = False) -> None:
    config = RAGConfig.from_env()
    if not config.rag_enabled:
        raise RuntimeError(
            "RAG is not configured. Set GEMINI_API_KEY, PINECONE_API_KEY, and PINECONE_INDEX in .env."
        )

    documents = collect_documents(DOCUMENTS_DIR)
    if not documents:
        raise RuntimeError(f"No documents found in {DOCUMENTS_DIR}")

    client = genai.Client(api_key=config.gemini_api_key)
    index = get_pinecone_index(config)

    if reset:
        clear_namespace(index, config.pinecone_namespace)
        print(f"Cleared namespace: {config.pinecone_namespace}")

    total_vectors = 0
    for document_path in documents:
        text = load_document(document_path)
        chunks = chunk_text(text, config.chunk_size, config.chunk_overlap)
        if not chunks:
            print(f"Skipped empty document: {document_path.name}")
            continue

        embeddings = embed_documents(client, config, chunks)
        source = document_path.name
        count = upsert_chunks(
            index=index,
            namespace=config.pinecone_namespace,
            chunks=chunks,
            embeddings=embeddings,
            source=source,
        )
        total_vectors += count
        print(f"Indexed {count} chunks from {source}")

    print(
        f"Done. Upserted {total_vectors} vectors into "
        f"{config.pinecone_index}/{config.pinecone_namespace}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all vectors in the namespace before ingesting",
    )
    args = parser.parse_args()
    ingest_documents(reset=args.reset)


if __name__ == "__main__":
    main()
