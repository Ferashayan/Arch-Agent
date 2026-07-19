"""
services/floor_plan_service.py
-------------------------------
Generates a 2D floor plan image from an architectural report using any
OpenAI-compatible image-generation model (default: wan2.7-image-pro via
DashScope's compatible-mode endpoint).

Because we use the standard OpenAI client, switching models is a one-line
change — just update FLOOR_PLAN_MODEL in your .env (or pass `model=` to the
function directly).

Public API
----------
generate_floor_plan(
    report_text    : str,
    api_key        : str | None  = None,   # falls back to DASHSCOPE_API_KEY
    model          : str | None  = None,   # falls back to FLOOR_PLAN_MODEL env / wan2.7-image-pro
    use_intl       : bool        = True,   # True = intl endpoint (outside China)
) -> bytes | None

Endpoints
---------
International : https://dashscope-intl.aliyuncs.com/compatible-mode/v1
China         : https://dashscope.aliyuncs.com/compatible-mode/v1
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the project root (works whether called from Streamlit or standalone)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — override any of these in .env
# ---------------------------------------------------------------------------
_DEFAULT_MODEL        = os.getenv("FLOOR_PLAN_MODEL", "wan2.7-image-pro")
_BASE_URL_INTL        = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
_BASE_URL_CHINA       = "https://dashscope.aliyuncs.com/compatible-mode/v1"


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
    Extract the image URL from an OpenAI-compatible chat completion response.
    Handles both plain-string content and list-of-content-items formats.
    """
    choices = response.choices
    if not choices:
        return None

    content = choices[0].message.content

    # Case 1: content is a plain URL string
    if isinstance(content, str) and content.startswith("http"):
        return content

    # Case 2: content is a list of typed items
    if isinstance(content, list):
        for item in content:
            # item may be a dict or a Pydantic model
            if isinstance(item, dict):
                t = item.get("type", "")
                if t == "image_url":
                    return item.get("image_url", {}).get("url")
                if t == "image":
                    return item.get("image")
                # plain URL value in any key
                for v in item.values():
                    if isinstance(v, str) and v.startswith("http"):
                        return v
            else:
                # Pydantic-style attribute access
                t = getattr(item, "type", "")
                if t == "image_url":
                    iu = getattr(item, "image_url", None)
                    return getattr(iu, "url", None)
                if t == "image":
                    return getattr(item, "image", None)

    return None


def generate_floor_plan(
    report_text: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    use_intl: bool = True,
) -> Optional[bytes]:
    """
    Calls an OpenAI-compatible image-generation model (default: wan2.7-image-pro)
    and returns the raw image bytes.

    Parameters
    ----------
    report_text : str
        Full architectural report produced by the main agent.
    api_key : str, optional
        API key. Falls back to DASHSCOPE_API_KEY from .env.
    model : str, optional
        Model name. Falls back to FLOOR_PLAN_MODEL env var or 'wan2.7-image-pro'.
    use_intl : bool
        True  -> dashscope-intl endpoint (outside China, default)
        False -> dashscope China endpoint

    Returns
    -------
    bytes or None
    """
    if not report_text:
        logger.warning("floor_plan_service: no report_text provided.")
        return None

    resolved_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
    if not resolved_key:
        raise RuntimeError(
            "No API key found. Set DASHSCOPE_API_KEY in .env or pass api_key= to the function."
        )

    resolved_model = model or _DEFAULT_MODEL
    base_url       = _BASE_URL_INTL if use_intl else _BASE_URL_CHINA

    logger.info(
        "floor_plan_service: model=%s  endpoint=%s  key_src=%s",
        resolved_model, base_url,
        "parameter" if api_key else "env"
    )

    client = OpenAI(api_key=resolved_key, base_url=base_url)

    response = client.chat.completions.create(
        model=resolved_model,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": _build_prompt(report_text)}],
            }
        ],
        n=1,
    )

    image_url = _extract_image_url(response)
    if not image_url:
        raise RuntimeError(
            f"No image URL found in response. Full response:\n{response}"
        )

    logger.info("floor_plan_service: downloading image from %s", image_url)
    with urlopen(image_url, timeout=30) as resp:
        return resp.read()
