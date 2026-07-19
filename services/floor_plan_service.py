"""
services/floor_plan_service.py
-------------------------------
Generates a 2D floor plan image from an architectural report using the
DashScope ImageGeneration API (Wan2.7-image-pro by default).

Switching models is a single-line change — just set FLOOR_PLAN_MODEL in .env
or pass model= to generate_floor_plan().

Public API
----------
generate_floor_plan(
    report_text : str,
    api_key     : str | None = None,   # falls back to DASHSCOPE_API_KEY in .env
    model       : str | None = None,   # falls back to FLOOR_PLAN_MODEL / wan2.7-image-pro
    use_intl    : bool       = True,   # True = international endpoint (outside China)
) -> bytes | None

Endpoints
---------
International : https://dashscope-intl.aliyuncs.com/api/v1  (default)
China         : https://dashscope.aliyuncs.com/api/v1
"""

from __future__ import annotations

import logging
import os
from http import HTTPStatus
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import dashscope
from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message
from dotenv import load_dotenv

# Load .env from project root so the service also works standalone
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — all overridable via .env or function arguments
# ---------------------------------------------------------------------------
_DEFAULT_MODEL  = os.getenv("FLOOR_PLAN_MODEL", "wan2.7-image")
_ENDPOINT_INTL  = "https://dashscope-intl.aliyuncs.com/api/v1"
_ENDPOINT_CHINA = "https://dashscope.aliyuncs.com/api/v1"


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


def _extract_image_url(response) -> Optional[str]:
    """
    Extract the image URL from a DashScope ImageGeneration response.
    Confirmed structure: output.choices[0].message.content[0].image
    """
    try:
        output  = response.output
        choices = output.get("choices", [])
        content = choices[0].get("message", {}).get("content", []) if choices else []
        for item in content:
            if item.get("type") == "image" and item.get("image"):
                return item["image"]
        # Fallback — try any http value in content items
        for item in content:
            for v in item.values():
                if isinstance(v, str) and v.startswith("http"):
                    return v
    except Exception as exc:
        logger.error("_extract_image_url parse error: %s", exc)
    return None


def generate_floor_plan(
    report_text: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    use_intl: bool = True,
) -> Optional[bytes]:
    """
    Calls DashScope ImageGeneration (Wan2.7-image-pro by default) and returns
    raw image bytes of the generated 2D floor plan.

    Parameters
    ----------
    report_text : str
        Full architectural report produced by the main agent.
    api_key : str, optional
        DashScope API key. Falls back to DASHSCOPE_API_KEY from .env.
    model : str, optional
        Model name. Falls back to FLOOR_PLAN_MODEL env var or 'wan2.7-image-pro'.
        Change this to switch models without editing code.
    use_intl : bool
        True  -> dashscope-intl.aliyuncs.com (outside China, default)
        False -> dashscope.aliyuncs.com       (inside China)

    Returns
    -------
    bytes or None
    """
    if not report_text:
        logger.warning("floor_plan_service: no report_text provided.")
        return None

    # Resolve key: passed argument wins, then fall back to .env
    resolved_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
    if not resolved_key:
        raise RuntimeError(
            "No DashScope API key found. "
            "Set DASHSCOPE_API_KEY in .env or pass api_key= to this function."
        )

    resolved_model = model or _DEFAULT_MODEL
    endpoint       = _ENDPOINT_INTL if use_intl else _ENDPOINT_CHINA

    # Apply global DashScope settings before calling
    dashscope.api_key          = resolved_key
    dashscope.base_http_api_url = endpoint

    logger.info(
        "floor_plan_service: model=%s | endpoint=%s | key_src=%s",
        resolved_model, endpoint,
        "parameter" if api_key else "DASHSCOPE_API_KEY env",
    )

    message = Message(
        role="user",
        content=[{"text": _build_prompt(report_text)}],
    )

    response = ImageGeneration.call(
        model=resolved_model,
        api_key=resolved_key,
        messages=[message],
        n=1,
        size="1024*1024",
    )

    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"DashScope call failed [{response.status_code}]: "
            f"{response.get('code', 'N/A')} -- {response.get('message', str(response))}"
        )

    image_url = _extract_image_url(response)
    if not image_url:
        raise RuntimeError(
            f"No image URL in DashScope response.\nFull response: {response}"
        )

    logger.info("floor_plan_service: downloading image from %s", image_url)
    with urlopen(image_url, timeout=30) as resp:
        return resp.read()
