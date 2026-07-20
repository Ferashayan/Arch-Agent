"""
api/routes/generate.py
----------------------
POST /api/generate/floor-plan   — 2D floor plan image
POST /api/generate/exterior     — 3D exterior design image
POST /api/generate/ifc          — IFC BIM model file
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from core.schemas import (
    GenerateExteriorRequest,
    GenerateFloorPlanRequest,
    GenerateIFCRequest,
    IFCResponse,
    ImageResponse,
    base64_to_bytes,
    bytes_to_base64,
)
from services.exterior_design_service import generate_exterior_design
from services.floor_plan_service import generate_floor_plan
from services.ifc_builder_service import generate_ifc_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/generate", tags=["Generation"])


@router.post("/floor-plan", response_model=ImageResponse)
async def generate_floor_plan_endpoint(request: GenerateFloorPlanRequest):
    """
    Generate a 2D architectural floor plan from the analysis report.
    Returns base64-encoded PNG image.
    """
    try:
        image_bytes = generate_floor_plan(report_text=request.report_text)

        if not image_bytes:
            raise HTTPException(
                status_code=502,
                detail="Gemini model did not return a floor plan image.",
            )

        # Detect mime type
        mime = "image/jpeg" if image_bytes.startswith(b"\xff\xd8") else "image/png"
        ext = "jpg" if mime == "image/jpeg" else "png"

        return ImageResponse(
            image_base64=bytes_to_base64(image_bytes),
            mime_type=mime,
            filename=f"floor_plan.{ext}",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Floor plan generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/exterior", response_model=ImageResponse)
async def generate_exterior_endpoint(request: GenerateExteriorRequest):
    """
    Generate a 3D photorealistic exterior design render.
    Optionally accepts the 2D floor plan image as base64 input.
    """
    try:
        floor_plan_bytes = None
        if request.floor_plan_base64:
            floor_plan_bytes = base64_to_bytes(request.floor_plan_base64)

        image_bytes = generate_exterior_design(
            report_text=request.report_text,
            floor_plan_bytes=floor_plan_bytes,
            style=request.style,
        )

        if not image_bytes:
            raise HTTPException(
                status_code=502,
                detail="Gemini model did not return an exterior design image.",
            )

        mime = "image/jpeg" if image_bytes.startswith(b"\xff\xd8") else "image/png"
        ext = "jpg" if mime == "image/jpeg" else "png"

        return ImageResponse(
            image_base64=bytes_to_base64(image_bytes),
            mime_type=mime,
            filename=f"exterior_design.{ext}",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Exterior design generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ifc", response_model=IFCResponse)
async def generate_ifc_endpoint(request: GenerateIFCRequest):
    """
    Generate an IFC BIM 3D model from the analysis report.
    Returns base64-encoded IFC file + extracted JSON coordinates.
    """
    try:
        floor_plan_bytes = None
        if request.floor_plan_base64:
            floor_plan_bytes = base64_to_bytes(request.floor_plan_base64)

        ifc_bytes, coords_data = generate_ifc_model(
            report_text=request.report_text,
            floor_plan_bytes=floor_plan_bytes,
        )

        if not ifc_bytes or not coords_data:
            raise HTTPException(
                status_code=502,
                detail="Failed to generate IFC model.",
            )

        summary = {
            "stories": len(coords_data.get("stories", [])),
            "walls": len(coords_data.get("walls", [])),
            "rooms": len(coords_data.get("rooms", [])),
            "openings": len(coords_data.get("openings", [])),
        }

        return IFCResponse(
            ifc_base64=bytes_to_base64(ifc_bytes),
            coords=coords_data,
            filename="architectural_villa_model.ifc",
            summary=summary,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("IFC model generation failed")
        raise HTTPException(status_code=500, detail=str(exc))
