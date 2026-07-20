"""
core/schemas.py
---------------
Pydantic v2 request / response models shared between the FastAPI routes
and the core engine.  These form the typed contract that a Next.js (or
any other) frontend can consume.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------
class LandDefaults(BaseModel):
    """Default sidebar values that complement the raw user input."""
    land_area: float = Field(400.0, description="Default land area in m² if not in JSON")
    street_width: float = Field(15.0, description="Default street width in meters")
    style: str = Field("مودرن حديث", description="Preferred architectural style")


class RAGSourceChunk(BaseModel):
    """A single chunk retrieved from Pinecone."""
    source: str = Field(..., description="Document filename")
    text: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Cosine similarity score")


class ParsedBoundary(BaseModel):
    length: float = 0.0
    street_width: float = 0.0
    desc: str = ""


# ---------------------------------------------------------------------------
# /api/analyze
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    """Payload for the main analysis endpoint."""
    raw_input: str = Field(..., description="Raw user input — either a JSON string (deed data) or plain Arabic text")
    land_area: float = Field(400.0, description="Default land area from UI sidebar")
    street_width: float = Field(15.0, description="Default street width from UI sidebar")
    style: str = Field("مودرن حديث", description="Preferred architectural style")


class AnalyzeResponse(BaseModel):
    """Full response from the analysis pipeline."""
    report: str = Field(..., description="Complete architectural analysis report (Arabic)")
    client_summary: str = Field("", description="Formatted client requirements block shown to user")
    rag_sources: List[RAGSourceChunk] = Field(default_factory=list, description="RAG source chunks used")
    is_json_input: bool = Field(False, description="Whether the input was parsed as JSON deed data")


# ---------------------------------------------------------------------------
# /api/generate/floor-plan
# ---------------------------------------------------------------------------
class GenerateFloorPlanRequest(BaseModel):
    report_text: str = Field(..., description="Architectural report text to base the floor plan on")


class ImageResponse(BaseModel):
    """Base64-encoded image response."""
    image_base64: str = Field(..., description="Base64-encoded image bytes")
    mime_type: str = Field("image/png", description="MIME type of the image")
    filename: str = Field("image.png", description="Suggested download filename")


# ---------------------------------------------------------------------------
# /api/generate/exterior
# ---------------------------------------------------------------------------
class GenerateExteriorRequest(BaseModel):
    report_text: str = Field(..., description="Architectural report text")
    floor_plan_base64: Optional[str] = Field(None, description="Base64-encoded 2D floor plan image (optional)")
    style: Optional[str] = Field(None, description="Architectural style preference")


# ---------------------------------------------------------------------------
# /api/generate/ifc
# ---------------------------------------------------------------------------
class GenerateIFCRequest(BaseModel):
    report_text: str = Field(..., description="Architectural report text")
    floor_plan_base64: Optional[str] = Field(None, description="Base64-encoded 2D floor plan image (optional)")


class IFCResponse(BaseModel):
    """IFC model generation response."""
    ifc_base64: str = Field(..., description="Base64-encoded IFC file bytes")
    coords: Dict[str, Any] = Field(default_factory=dict, description="Extracted JSON coordinates data")
    filename: str = Field("architectural_villa_model.ifc", description="Suggested download filename")
    summary: Dict[str, int] = Field(default_factory=dict, description="Element counts (stories, walls, rooms, openings)")


# ---------------------------------------------------------------------------
# /api/ingest
# ---------------------------------------------------------------------------
class IngestRequest(BaseModel):
    reset: bool = Field(False, description="Whether to clear the namespace before ingesting")


class IngestResponse(BaseModel):
    total_vectors: int = Field(..., description="Total vectors upserted")
    message: str = Field(..., description="Status message")


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field("ok")
    gemini_model: str = Field("")
    rag_enabled: bool = Field(False)
    floor_plan_model: str = Field("")
    exterior_model: str = Field("")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def bytes_to_base64(data: bytes) -> str:
    """Encode raw bytes to a base64 string."""
    return base64.b64encode(data).decode("utf-8")


def base64_to_bytes(b64: str) -> bytes:
    """Decode a base64 string back to raw bytes."""
    return base64.b64decode(b64)
