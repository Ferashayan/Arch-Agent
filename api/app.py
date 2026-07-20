"""
api/app.py
----------
FastAPI application factory.
Configures CORS, mounts route modules, and provides the /api/health endpoint.

Usage:
    uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.schemas import HealthResponse
from rag.config import RAGConfig

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate configuration on startup."""
    config = RAGConfig.from_env()
    api_key = os.getenv("GEMINI_API_KEY", "")

    logger.info("=" * 60)
    logger.info("Arch-Agent FastAPI Server Starting")
    logger.info("  Gemini Model   : %s", config.gemini_model)
    logger.info("  RAG Enabled    : %s", config.rag_enabled)
    logger.info("  Gemini API Key : %s", "✓ Set" if api_key else "✗ Missing")
    logger.info("  Pinecone Index : %s", config.pinecone_index)
    logger.info("=" * 60)

    if not api_key:
        logger.warning(
            "GEMINI_API_KEY is not set. Analysis and generation endpoints will fail."
        )

    yield  # Application runs here

    logger.info("Arch-Agent FastAPI Server Shutting Down")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="Arch-Agent API",
        description=(
            "المساعد المعماري السعودي — FastAPI backend for architectural analysis, "
            "floor plan generation, exterior design, and IFC BIM model creation."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────
    cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────
    from api.routes.analyze import router as analyze_router
    from api.routes.generate import router as generate_router
    from api.routes.ingest import router as ingest_router

    app.include_router(analyze_router)
    app.include_router(generate_router)
    app.include_router(ingest_router)

    # ── Health Check ──────────────────────────────────────────
    @app.get("/api/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        config = RAGConfig.from_env()
        return HealthResponse(
            status="ok",
            gemini_model=config.gemini_model,
            rag_enabled=config.rag_enabled,
            floor_plan_model=os.getenv("FLOOR_PLAN_MODEL", "gemini-3.1-flash-image"),
            exterior_model=os.getenv("EXTERIOR_DESIGN_MODEL", "gemini-3.1-flash-image"),
        )

    return app


# Singleton instance for `uvicorn api.app:app`
app = create_app()
