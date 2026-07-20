"""
services/ifc_builder_service.py
--------------------------------
Generates an IFC (Industry Foundation Classes) 3D BIM architectural model from
an architectural report (and optional 2D floor plan image) using Gemini for
layout JSON coordinate extraction and IfcOpenShell for IFC construction.

Public API
----------
generate_ifc_model(
    report_text      : str,
    floor_plan_bytes : bytes | None = None,
    api_key          : str | None   = None,
    model            : str | None   = None,
    output_path      : str | Path | None = None,
) -> tuple[bytes | None, dict | None]
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

import ifcopenshell
import ifcopenshell.api
import numpy as np

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


# ---------------------------------------------------------------------------
# 1. Deterministic Pydantic Schemas with Explicit Scalar Coordinate Fields
# ---------------------------------------------------------------------------
class BuildingModel(BaseModel):
    length_m: float = Field(15.0, description="Building total length in meters")
    width_m: float = Field(12.0, description="Building total width in meters")


class StoryModel(BaseModel):
    name: str = Field("Ground Floor", description="Floor name")
    elevation_m: float = Field(0.0, description="Elevation level from ground in meters")
    height_m: float = Field(3.5, description="Clear floor height in meters")


class RoomModel(BaseModel):
    name: str = Field(..., description="Room or space label")
    story: str = Field("Ground Floor", description="Associated floor name")
    x: float = Field(..., description="X coordinate of room bottom-left corner")
    y: float = Field(..., description="Y coordinate of room bottom-left corner")
    width_m: float = Field(..., description="Room width along X axis")
    length_m: float = Field(..., description="Room length along Y axis")


class WallModel(BaseModel):
    name: str = Field("Wall", description="Wall label")
    story: str = Field("Ground Floor", description="Associated floor name")
    start_x: float = Field(..., description="Wall start X coordinate")
    start_y: float = Field(..., description="Wall start Y coordinate")
    end_x: float = Field(..., description="Wall end X coordinate")
    end_y: float = Field(..., description="Wall end Y coordinate")
    height_m: float = Field(3.5, description="Wall height in meters")
    thickness_m: float = Field(0.3, description="Wall thickness in meters")


class SlabModel(BaseModel):
    name: str = Field("Slab", description="Slab label")
    story: str = Field("Ground Floor", description="Associated floor name")
    thickness_m: float = Field(0.25, description="Slab thickness in meters")


class OpeningModel(BaseModel):
    name: str = Field("Opening", description="Door or Window label")
    type: str = Field("door", description="'door' or 'window'")
    story: str = Field("Ground Floor", description="Associated floor name")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    z: float = Field(0.0, description="Z offset from floor level")
    width_m: float = Field(1.0, description="Opening width")
    height_m: float = Field(2.1, description="Opening height")


class IFCLayoutSchema(BaseModel):
    project_name: str = Field("Saudi Villa Architectural Project", description="Project title")
    building: BuildingModel
    stories: List[StoryModel]
    rooms: List[RoomModel]
    walls: List[WallModel]
    slabs: List[SlabModel]
    openings: List[OpeningModel]


# ---------------------------------------------------------------------------
# 2. Clean Prompt Builder (No raw JSON templates that conflict with Pydantic)
# ---------------------------------------------------------------------------
def _build_prompt(report_text: str) -> str:
    return (
        "You are an expert BIM structural engineer. Analyze the architectural report below "
        "(and any provided 2D floor plan drawing) and extract exact 3D building element coordinates "
        "and spatial dimensions required to construct a 3D IFC building model.\n\n"
        "Extraction Requirements:\n"
        "1. Identify overall building dimensions (length_m, width_m).\n"
        "2. List all floors/stories with elevation_m and height_m.\n"
        "3. Extract room spaces with ground-plane coordinates (x, y, width_m, length_m).\n"
        "4. Extract perimeter and interior walls with 2D start points (start_x, start_y) and end points (end_x, end_y), plus height_m and thickness_m.\n"
        "5. Extract floor slabs for each story.\n"
        "6. Extract doors and windows with position (x, y, z) and dimensions (width_m, height_m).\n\n"
        "Architectural Report (Arabic):\n"
        "---\n"
        f"{report_text}\n"
        "---\n"
    )


def _clean_and_parse_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    # Fix common LLM json formatting issues (e.g. trailing commas)
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# 3. IFC Model Construction via IfcOpenShell
# ---------------------------------------------------------------------------
def build_ifc_from_json(data: dict) -> bytes:
    """
    Constructs a valid IFC4 3D BIM model using IfcOpenShell based on extracted JSON coordinates.
    """
    f = ifcopenshell.api.run("project.create_file", version="IFC4")

    # Project setup
    project_name = data.get("project_name", "Architectural Villa Project")
    project = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcProject", name=project_name)

    # Contexts
    model_ctx = ifcopenshell.api.run("context.add_context", f, context_type="Model")
    body_ctx = ifcopenshell.api.run(
        "context.add_context",
        f,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=model_ctx,
    )

    # Spatial Hierarchy: Site and Building
    site = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcSite", name="Property Site")
    building = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuilding", name="Villa")

    ifcopenshell.api.run("aggregate.assign_object", f, products=[site], relating_object=project)
    ifcopenshell.api.run("aggregate.assign_object", f, products=[building], relating_object=site)

    # Stories
    stories_data = data.get("stories", [
        {"name": "Ground Floor", "elevation_m": 0.0, "height_m": 3.5},
        {"name": "First Floor", "elevation_m": 3.5, "height_m": 3.2},
    ])

    story_map: dict[str, tuple[ifcopenshell.entity_instance, float]] = {}
    for st_info in stories_data:
        st_name = st_info.get("name", "Storey")
        st_elev = float(st_info.get("elevation_m", 0.0))
        storey = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name=st_name)
        ifcopenshell.api.run("aggregate.assign_object", f, products=[storey], relating_object=building)
        story_map[st_name] = (storey, st_elev)

    default_story = list(story_map.values())[0][0] if story_map else ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name="Ground Floor")
    default_elev = list(story_map.values())[0][1] if story_map else 0.0

    # Rooms / Spaces (In IFC4 spaces use aggregate.assign_object to storey)
    rooms_data = data.get("rooms", [])
    for room_info in rooms_data:
        r_name = room_info.get("name", "Space")
        st_name = room_info.get("story", "Ground Floor")
        storey_ent, _ = story_map.get(st_name, (default_story, default_elev))

        space = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcSpace", name=r_name)
        ifcopenshell.api.run("aggregate.assign_object", f, products=[space], relating_object=storey_ent)

    # Walls
    walls_data = data.get("walls", [])
    for wall_info in walls_data:
        w_name = wall_info.get("name", "Wall")
        st_name = wall_info.get("story", "Ground Floor")
        storey_ent, st_elev = story_map.get(st_name, (default_story, default_elev))

        if "start_x" in wall_info and "start_y" in wall_info:
            x1, y1 = float(wall_info["start_x"]), float(wall_info["start_y"])
        else:
            start = wall_info.get("start", [0.0, 0.0])
            x1, y1 = float(start[0]), float(start[1])

        if "end_x" in wall_info and "end_y" in wall_info:
            x2, y2 = float(wall_info["end_x"]), float(wall_info["end_y"])
        else:
            end = wall_info.get("end", [5.0, 0.0])
            x2, y2 = float(end[0]), float(end[1])

        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length <= 0.01:
            length = 5.0
            angle = 0.0
        else:
            angle = math.atan2(dy, dx)

        height = float(wall_info.get("height_m", 3.5))
        thickness = float(wall_info.get("thickness_m", 0.3))

        wall = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcWall", name=w_name)
        ifcopenshell.api.run("spatial.assign_container", f, products=[wall], relating_structure=storey_ent)

        w_repr = ifcopenshell.api.run(
            "geometry.add_wall_representation",
            f,
            context=body_ctx,
            length=length,
            height=height,
            thickness=thickness,
        )
        ifcopenshell.api.run("geometry.assign_representation", f, product=wall, representation=w_repr)

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        matrix = np.eye(4)
        matrix[0, 0] = cos_a
        matrix[0, 1] = -sin_a
        matrix[1, 0] = sin_a
        matrix[1, 1] = cos_a
        matrix[0, 3] = x1
        matrix[1, 3] = y1
        matrix[2, 3] = st_elev

        ifcopenshell.api.run("geometry.edit_object_placement", f, product=wall, matrix=matrix)

    # Slabs
    slabs_data = data.get("slabs", [])
    b_len = float(data.get("building", {}).get("length_m", 15.0))
    b_wid = float(data.get("building", {}).get("width_m", 12.0))
    for slab_info in slabs_data:
        s_name = slab_info.get("name", "Slab")
        st_name = slab_info.get("story", "Ground Floor")
        storey_ent, st_elev = story_map.get(st_name, (default_story, default_elev))
        thickness = float(slab_info.get("thickness_m", 0.25))

        slab = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcSlab", name=s_name)
        ifcopenshell.api.run("spatial.assign_container", f, products=[slab], relating_structure=storey_ent)

        s_repr = ifcopenshell.api.run(
            "geometry.add_wall_representation",
            f,
            context=body_ctx,
            length=b_len,
            height=thickness,
            thickness=b_wid,
        )
        ifcopenshell.api.run("geometry.assign_representation", f, product=slab, representation=s_repr)

        matrix = np.eye(4)
        matrix[2, 3] = st_elev
        ifcopenshell.api.run("geometry.edit_object_placement", f, product=slab, matrix=matrix)

    # Openings (Doors & Windows)
    openings_data = data.get("openings", [])
    for op_info in openings_data:
        op_name = op_info.get("name", "Opening")
        op_type = str(op_info.get("type", "door")).lower()
        st_name = op_info.get("story", "Ground Floor")
        storey_ent, st_elev = story_map.get(st_name, (default_story, default_elev))

        if "x" in op_info and "y" in op_info:
            px, py = float(op_info["x"]), float(op_info["y"])
            pz = float(op_info.get("z", 0.0))
        else:
            pos = op_info.get("position", [2.0, 0.0, 0.0])
            px = float(pos[0]) if len(pos) > 0 else 2.0
            py = float(pos[1]) if len(pos) > 1 else 0.0
            pz = float(pos[2]) if len(pos) > 2 else 0.0

        w_val = float(op_info.get("width_m", 1.0))
        h_val = float(op_info.get("height_m", 2.1))

        if "door" in op_type:
            entity = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcDoor", name=op_name)
            op_repr = ifcopenshell.api.run(
                "geometry.add_door_representation",
                f,
                context=body_ctx,
                overall_width=w_val,
                overall_height=h_val,
            )
        else:
            entity = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcWindow", name=op_name)
            op_repr = ifcopenshell.api.run(
                "geometry.add_window_representation",
                f,
                context=body_ctx,
                overall_width=w_val,
                overall_height=h_val,
            )

        ifcopenshell.api.run("spatial.assign_container", f, products=[entity], relating_structure=storey_ent)
        ifcopenshell.api.run("geometry.assign_representation", f, product=entity, representation=op_repr)

        matrix = np.eye(4)
        matrix[0, 3] = px
        matrix[1, 3] = py
        matrix[2, 3] = st_elev + pz
        ifcopenshell.api.run("geometry.edit_object_placement", f, product=entity, matrix=matrix)

    return f.to_string().encode("utf-8")


# ---------------------------------------------------------------------------
# 4. Public API Function with Resilient Multimodal / Schema Fallbacks
# ---------------------------------------------------------------------------
def generate_ifc_model(
    report_text: str,
    floor_plan_bytes: Optional[bytes] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> Tuple[Optional[bytes], Optional[dict]]:
    """
    Analyzes architectural report & 2D floor plan using Gemini to obtain layout JSON coordinates,
    then constructs a fully structured IFC4 3D BIM model using IfcOpenShell.

    Returns
    -------
    tuple(ifc_bytes, coords_json)
    """
    if not report_text:
        logger.warning("ifc_builder_service: no report_text provided.")
        return None, None

    resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not resolved_key:
        raise RuntimeError(
            "No Gemini API key found. "
            "Set GEMINI_API_KEY in .env or pass api_key= to this function."
        )

    contents: list[str | types.Part] = [_build_prompt(report_text)]
    if floor_plan_bytes:
        mime_type = "image/jpeg" if floor_plan_bytes.startswith(b"\xff\xd8") else "image/png"
        contents.append(types.Part.from_bytes(data=floor_plan_bytes, mime_type=mime_type))

    models_to_try = [model] if model else [_DEFAULT_MODEL, "gemini-3.1-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash-lite"]
    client = genai.Client(api_key=resolved_key)
    coords_data: Optional[dict] = None

    for candidate_model in models_to_try:
        logger.info("ifc_builder_service: attempting model %s", candidate_model)
        
        # Strategy A: Try Structured Output with Pydantic Schema
        try:
            response = client.models.generate_content(
                model=candidate_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=IFCLayoutSchema,
                ),
            )
            if response and response.text:
                coords_data = json.loads(response.text)
                logger.info("ifc_builder_service: successfully parsed JSON using schema on model %s", candidate_model)
                break
        except Exception as schema_exc:
            logger.warning(
                "ifc_builder_service: Pydantic schema generation failed on model %s (%s). Attempting JSON prompt fallback...",
                candidate_model,
                schema_exc,
            )

        # Strategy B: Fallback for Multimodal / Image inputs without rigid schema constraint
        try:
            response = client.models.generate_content(
                model=candidate_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            if response and response.text:
                coords_data = _clean_and_parse_json(response.text)
                logger.info("ifc_builder_service: successfully parsed JSON using fallback parser on model %s", candidate_model)
                break
        except Exception as raw_exc:
            logger.warning("ifc_builder_service: raw JSON generation failed on model %s (%s).", candidate_model, raw_exc)

    if not coords_data:
        raise RuntimeError("Failed to extract valid 3D layout JSON coordinates from Gemini models.")

    logger.info("ifc_builder_service: constructing 3D IFC model via IfcOpenShell...")
    ifc_bytes = build_ifc_from_json(coords_data)

    if output_path:
        out_p = Path(output_path)
        out_p.write_bytes(ifc_bytes)
        logger.info("ifc_builder_service: saved IFC file to %s", out_p.resolve())

    return ifc_bytes, coords_data
