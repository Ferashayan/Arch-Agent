def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks
