"""
services/floor_plan_service.py
-------------------------------
Generates a 2D floor plan image from an architectural report using the
Google GenAI API (Gemini 3.1 Flash Image by default).

Switching models is a single-line change — just set FLOOR_PLAN_MODEL in .env
or pass model= to generate_floor_plan().

Public API
----------
generate_floor_plan(
    report_text : str,
    api_key     : str | None = None,   # falls back to GEMINI_API_KEY in .env
    model       : str | None = None,   # falls back to FLOOR_PLAN_MODEL / gemini-3.1-flash-image
) -> bytes | None
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env from project root so the service also works standalone
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — all overridable via .env or function arguments
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = os.getenv("FLOOR_PLAN_MODEL", "gemini-3.1-flash-image")


def _build_prompt(report_text: str) -> str:
    return (
        "Generate a clean, professional top-down 2D architectural floor plan drawing "
        "based on the following Arabic architectural report.\n\n"
        "Requirements:\n"
        "- Style: Minimal black-and-white technical line drawing on a pure white background.\n"
        "- Show: thick black walls, room labels in English, door arcs, window hatches, "
        "staircase stepped symbol, elevator shaft box.\n"
        "- Include ALL rooms and spaces mentioned in the report.\n"
        "- Organise as clearly labelled sections: Ground Floor, First Floor, Roof/Annex.\n"
        "- Standard architectural symbols only. No furniture, no colors, no shadows.\n"
        "- Proportions realistic to the land area and setbacks described.\n"
        "- North-arrow indicator in one corner.\n\n"
        "Architectural Report (Arabic):\n"
        "---\n"
        f"{report_text}\n"
        "---\n\n"
        "Output: a single clean 2D floor plan image. No text captions, no borders."
    )


def generate_floor_plan(
    report_text: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> Optional[bytes]:
    """
    Calls Google GenAI API (gemini-3.1-flash-image by default) and returns
    raw image bytes of the generated 2D floor plan.

    Parameters
    ----------
    report_text : str
        Full architectural report produced by the main agent.
    api_key : str, optional
        Gemini API key. Falls back to GEMINI_API_KEY from .env.
    model : str, optional
        Model name. Falls back to FLOOR_PLAN_MODEL env var or 'gemini-3.1-flash-image'.

    Returns
    -------
    bytes or None
    """
    if not report_text:
        logger.warning("floor_plan_service: no report_text provided.")
        return None

    # Resolve key: passed argument wins, then fall back to GEMINI_API_KEY from .env
    resolved_key = api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("DASHSCOPE_API_KEY", "")
    if not resolved_key:
        raise RuntimeError(
            "No Gemini API key found. "
            "Set GEMINI_API_KEY in .env or pass api_key= to this function."
        )

    resolved_model = model or _DEFAULT_MODEL

    logger.info(
        "floor_plan_service: model=%s | key_src=%s",
        resolved_model,
        "parameter" if api_key else "GEMINI_API_KEY env",
    )

    client = genai.Client(api_key=resolved_key)
    response = client.models.generate_content(
        model=resolved_model,
        contents=_build_prompt(report_text),
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )

    if response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                return part.inline_data.data

    raise RuntimeError("No image returned from Gemini model generation response.")

