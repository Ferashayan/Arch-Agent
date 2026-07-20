"""
api/routes/analyze.py
---------------------
POST /api/analyze        — Full analysis pipeline (sync)
POST /api/analyze/stream — SSE streaming variant
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.engine import (
    analyze_stream,
    build_system_prompt,
    parse_client_input,
    retrieve_rag_context,
)
from core.schemas import AnalyzeRequest, AnalyzeResponse, RAGSourceChunk

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Full analysis pipeline:
    1. Parse client input (JSON deed or plain text)
    2. Retrieve RAG context from Pinecone
    3. Build system prompt
    4. Call Gemini for full response
    """
    try:
        from core.engine import analyze_full

        # Step 1: Parse
        client_data = parse_client_input(
            raw_input=request.raw_input,
            default_land_area=request.land_area,
            default_street_width=request.street_width,
            default_style=request.style,
        )

        # Step 2: RAG
        rag_context, rag_chunks = retrieve_rag_context(client_data.rag_query)

        # Step 3: Prompt
        prompt = build_system_prompt(client_data.client_requirements_block, rag_context)

        # Step 4: Gemini
        report = analyze_full(prompt)

        # Build response
        sources = [
            RAGSourceChunk(source=c.source, text=c.text, score=c.score)
            for c in rag_chunks
        ]

        return AnalyzeResponse(
            report=report,
            client_summary=client_data.client_requirements_block,
            rag_sources=sources,
            is_json_input=client_data.is_json_input,
        )

    except Exception as exc:
        logger.exception("Analysis pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/analyze/stream")
async def analyze_stream_endpoint(request: AnalyzeRequest):
    """
    SSE streaming variant of the analysis pipeline.
    Streams Gemini tokens as `text/event-stream` Server-Sent Events.

    Event format:
        data: {"type": "meta", "client_summary": "...", "rag_sources": [...]}

        data: {"type": "chunk", "text": "..."}
        data: {"type": "chunk", "text": "..."}

        data: {"type": "done"}
    """
    try:
        # Step 1: Parse
        client_data = parse_client_input(
            raw_input=request.raw_input,
            default_land_area=request.land_area,
            default_street_width=request.street_width,
            default_style=request.style,
        )

        # Step 2: RAG
        rag_context, rag_chunks = retrieve_rag_context(client_data.rag_query)

        # Step 3: Prompt
        prompt = build_system_prompt(client_data.client_requirements_block, rag_context)

    except Exception as exc:
        logger.exception("Pre-stream pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))

    # Build metadata event
    sources = [
        {"source": c.source, "text": c.text, "score": c.score}
        for c in rag_chunks
    ]
    meta_event = json.dumps(
        {
            "type": "meta",
            "client_summary": client_data.client_requirements_block,
            "rag_sources": sources,
            "is_json_input": client_data.is_json_input,
        },
        ensure_ascii=False,
    )

    async def event_generator():
        # Emit metadata first
        yield f"data: {meta_event}\n\n"

        # Stream Gemini chunks
        try:
            async for chunk_text in analyze_stream(prompt):
                chunk_event = json.dumps(
                    {"type": "chunk", "text": chunk_text},
                    ensure_ascii=False,
                )
                yield f"data: {chunk_event}\n\n"

            # Signal completion
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as exc:
            error_event = json.dumps(
                {"type": "error", "message": str(exc)},
                ensure_ascii=False,
            )
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
