"""
services/floor_plan_service.py
-------------------------------
Sends the architectural report to Alibaba DashScope Wan2.7-image-pro to
generate a 2D floor plan image.

Public API
----------
generate_floor_plan(report_text, dashscope_api_key, use_intl_endpoint=True)
    -> bytes | None

Notes
-----
- Uses ImageGeneration.call() (synchronous) from dashscope.aigc.image_generation
- International endpoint (outside China): dashscope-intl.aliyuncs.com/api/v1
- China endpoint: dashscope.aliyuncs.com/api/v1
"""

from __future__ import annotations

import logging
import os
from http import HTTPStatus
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

from dotenv import load_dotenv
import dashscope
from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message

# Load .env from the project root so the service works standalone too
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

_WAN_MODEL        = "wan2.7-image-pro"
_ENDPOINT_INTL    = "https://dashscope-intl.aliyuncs.com/api/v1"
_ENDPOINT_CHINA   = "https://dashscope.aliyuncs.com/api/v1"


def _build_prompt(report_text: str) -> str:
    return (
        "Generate a clean, professional top-down 2D architectural floor plan drawing "
        "based on the following Arabic architectural report.\n\n"
        "Requirements:\n"
        "- Style: Minimal black-and-white technical line drawing on a pure white background.\n"
        "- Show: thick black walls, room labels in English, door arcs, window hatches, "
        "staircase stepped symbol, elevator shaft box.\n"
        "- Include ALL rooms and spaces mentioned in the report.\n"
        "- Organise as separate clearly labelled sections: Ground Floor, First Floor, Roof/Annex.\n"
        "- Standard architectural symbols only. No furniture, no colors, no shadows.\n"
        "- Proportions realistic to the land area and setbacks described.\n"
        "- North-arrow indicator in one corner.\n\n"
        "Architectural Report (Arabic):\n"
        "---\n"
        f"{report_text}\n"
        "---\n\n"
        "Output: a single clean 2D floor plan image. No text captions, no decorative borders."
    )


def generate_floor_plan(
    report_text: str,
    dashscope_api_key: str,
    use_intl_endpoint: bool = True,
) -> Optional[bytes]:
    """
    Calls Wan2.7-image-pro via DashScope ImageGeneration API and returns
    the raw image bytes.

    Parameters
    ----------
    report_text : str
        Full architectural report from the main agent.
    dashscope_api_key : str
        Alibaba DashScope API key.
    use_intl_endpoint : bool
        True  -> dashscope-intl.aliyuncs.com  (outside China, default)
        False -> dashscope.aliyuncs.com        (inside China)

    Returns
    -------
    bytes or None
    """
    if not report_text:
        logger.warning("floor_plan_service: missing report_text.")
        return None

    # Resolve the API key: prefer the passed argument, then fall back to env
    resolved_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")
    if not resolved_key:
        raise RuntimeError(
            "DashScope API key not found. Set DASHSCOPE_API_KEY in .env or provide it in the sidebar."
        )
    logger.info("floor_plan_service: using key source=%s",
                "parameter" if dashscope_api_key else "DASHSCOPE_API_KEY env")

    # Set endpoint and key BEFORE the call
    dashscope.base_http_api_url = _ENDPOINT_INTL if use_intl_endpoint else _ENDPOINT_CHINA
    logger.info("floor_plan_service: endpoint=%s", dashscope.base_http_api_url)

    message = Message(
        role="user",
        content=[{"text": _build_prompt(report_text)}],
    )

    response = ImageGeneration.call(
        model=_WAN_MODEL,
        api_key=resolved_key,
        messages=[message],
        n=1,
        size="1024*1024",
    )

    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"DashScope submission failed [{response.status_code}]: "
            f"{response.get('code', 'N/A')} -- {response.get('message', str(response))}"
        )

    # Parse response — actual structure confirmed by live test:
    # output.choices[0].message.content[0].image  →  the image URL
    try:
        output   = response.output
        choices  = output.get("choices", [])
        content  = choices[0].get("message", {}).get("content", []) if choices else []
        # Primary path: content item with type="image" has key "image"
        image_url = None
        for item in content:
            if item.get("type") == "image" and item.get("image"):
                image_url = item["image"]
                break
        # Fallback: other possible key names
        if not image_url and content:
            first = content[0]
            image_url = (
                first.get("image")
                or first.get("image_url")
                or first.get("url")
            )
    except Exception as parse_err:
        raise RuntimeError(
            f"Could not parse DashScope response: {parse_err}\nFull response: {response}"
        )

    if not image_url:
        raise RuntimeError(
            f"No image URL found in DashScope response. Full response: {response}"
        )

    logger.info("floor_plan_service: downloading image from %s", image_url)
    with urlopen(image_url, timeout=30) as resp:
        return resp.read()
