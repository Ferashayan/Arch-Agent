"""
core/engine.py
--------------
Pure-Python business logic extracted from the monolithic main.py Streamlit app.
No Streamlit imports — this module is framework-agnostic and can be consumed
by FastAPI, CLI tools, or any other frontend.

Public Functions
----------------
- parse_client_input(raw_input, defaults) -> ParsedClientData
- build_system_prompt(client_data, rag_context) -> str
- analyze_stream(prompt, api_key, model) -> AsyncGenerator[str, None]
- analyze_full(prompt, api_key, model) -> str
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from google import genai

from rag.config import RAGConfig
from rag.retriever import RetrievedChunk, retrieve_context

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers (internal, richer than the Pydantic request schemas)
# ---------------------------------------------------------------------------
@dataclass
class ParsedBoundary:
    length: float = 0.0
    street_width: float = 0.0
    desc: str = ""


@dataclass
class ParsedClientData:
    """All data extracted from the user's input + sidebar defaults."""

    is_json_input: bool = False
    client_requirements_block: str = ""
    rag_query: str = ""

    # Only populated when is_json_input is True
    doc_number: str = ""
    doc_type: str = ""
    operation_type: str = ""
    owner_names: list[str] = field(default_factory=list)
    city: str = ""
    district: str = ""
    plan_number: str = ""
    parcel_number: str = ""
    land_area: float = 400.0
    street_width: float = 15.0
    style: str = "مودرن حديث"
    boundaries: dict[str, ParsedBoundary] = field(default_factory=dict)
    estimated_family_count: int = 4
    has_elderly: bool = False
    bedrooms_count: int = 3
    master_count: int = 1
    guest_reception_label: str = ""
    kitchen_label: str = ""
    has_maid: bool = False
    has_driver: bool = False
    has_laundry: bool = False
    has_storage: bool = False
    restrictions: str = "لا يوجد"


# ---------------------------------------------------------------------------
# 1. Parse Client Input
# ---------------------------------------------------------------------------
def _parse_street_width(boundary_data: dict) -> ParsedBoundary:
    """Extract numeric street width from boundary description text."""
    desc = boundary_data.get("description", "")
    length = boundary_data.get("length_m", 0.0)
    match = re.search(r"عرض\s*(\d+(?:\.\d+)?)", desc)
    width = float(match.group(1)) if match else 0.0
    return ParsedBoundary(length=length, street_width=width, desc=desc)


def parse_client_input(
    raw_input: str,
    default_land_area: float = 400.0,
    default_street_width: float = 15.0,
    default_style: str = "مودرن حديث",
) -> ParsedClientData:
    """
    Parse raw user input (JSON deed or plain text) into structured data.
    Mirrors the parsing logic from main.py sections 2-3.
    """
    data = ParsedClientData(
        land_area=default_land_area,
        street_width=default_street_width,
        style=default_style,
    )

    # Attempt JSON parsing
    try:
        parsed_json = json.loads(raw_input)
        data.is_json_input = True
    except json.JSONDecodeError:
        data.is_json_input = False
        data.client_requirements_block = f'طلب المستخدم المباشر: "{raw_input}"'
        data.rag_query = raw_input
        return data

    # --- JSON deed parsing ---

    # أ) Document details
    doc_data = parsed_json.get("document", {})
    data.doc_number = doc_data.get("document_number", "")
    data.doc_type = doc_data.get("document_type", "")
    data.operation_type = doc_data.get("operation_type", "")

    owners = parsed_json.get("owners", [])
    data.owner_names = [owner.get("name", "") for owner in owners]

    # ب) Property info
    property_info = parsed_json.get("property", {})
    data.city = property_info.get("city", "")
    data.district = property_info.get("district", "")
    data.plan_number = property_info.get("plan_number", "")
    data.parcel_number = property_info.get("parcel_number", "")
    data.land_area = property_info.get("area_m2", default_land_area)

    # ج) Boundaries
    boundaries_raw = parsed_json.get("land_details", {}).get("boundaries", {})
    data.boundaries = {
        "north": _parse_street_width(boundaries_raw.get("north", {})),
        "south": _parse_street_width(boundaries_raw.get("south", {})),
        "east": _parse_street_width(boundaries_raw.get("east", {})),
        "west": _parse_street_width(boundaries_raw.get("west", {})),
    }

    all_widths = [b.street_width for b in data.boundaries.values()]
    if any(all_widths):
        data.street_width = max(all_widths)

    # د) Family preferences
    family_prefs = parsed_json.get("family_preferences", {})
    data.estimated_family_count = family_prefs.get("family_members", {}).get("estimated_count", 4)
    data.has_elderly = family_prefs.get("elderly_or_accessibility", {}).get("has_elderly", False)
    data.bedrooms_count = family_prefs.get("bedrooms", {}).get("count", 3)
    data.master_count = family_prefs.get("master_bedrooms", {}).get("count", 1)
    data.guest_reception_label = family_prefs.get("guest_reception", {}).get("label", "")
    data.kitchen_label = family_prefs.get("kitchen", {}).get("label", "")

    additional_rooms = family_prefs.get("additional_rooms", {})
    data.has_maid = additional_rooms.get("maid_room", False)
    data.has_driver = additional_rooms.get("driver_room", False)
    data.has_laundry = additional_rooms.get("laundry_room", False)
    data.has_storage = additional_rooms.get("storage_room", False)

    data.restrictions = parsed_json.get("extra_information", {}).get("restrictions", "لا يوجد")

    # Build formatted requirements block
    b = data.boundaries
    data.client_requirements_block = f"""
### 📋 البيانات المستخرجة من الصك والطلب بنجاح:
* **وثيقة العقار:** {data.doc_type} رقم `{data.doc_number}` ({data.operation_type})
* **الموقع:** مدينة {data.city}، حي {data.district} (مخطط رقم: {data.plan_number})
* **التصنيف النطاقي:** {data.parcel_number}
* **مساحة الأرض المعتمدة:** {data.land_area} متر مربع
* **حدود وعروض الشوارع المحيطة للأرض:**
    * **شمالاً:** بطول {b['north'].length}م ({b['north'].desc})
    * **جنوباً:** بطول {b['south'].length}م ({b['south'].desc})
    * **شرقاً:** بطول {b['east'].length}م ({b['east'].desc})
    * **غرباً:** بطول {b['west'].length}م ({b['west'].desc})
* **تفاصيل العائلة والاحتياجات الممررة:**
    * عدد الأفراد: {data.estimated_family_count} أشخاص | كبار سن/احتياجات خاصة: {"نعم" if data.has_elderly else "لا"}
    * غرف النوم: {data.bedrooms_count} (منها غرف ماستر: {data.master_count})
    * نمط الاستقبال: {data.guest_reception_label} | نمط المطبخ: {data.kitchen_label}
    * غرف خدمية إضافية: عاملة منزليّة ({"نعم" if data.has_maid else "لا"})، سائق ({"نعم" if data.has_driver else "لا"})، غسيل ({"نعم" if data.has_laundry else "لا"})، مستودع ({"نعم" if data.has_storage else "لا"})
* **قيود إضافية:** {data.restrictions}
"""

    data.rag_query = (
        f"اشتراطات كود البناء أمانة مدينة {data.city} حي {data.district} "
        f"تصنيف {data.parcel_number} شوارع عرض "
        f"{[w for w in all_widths if w > 0]}"
    )

    return data


# ---------------------------------------------------------------------------
# 2. RAG Context Retrieval (thin wrapper)
# ---------------------------------------------------------------------------
def retrieve_rag_context(
    query: str,
    config: Optional[RAGConfig] = None,
) -> tuple[str, list[RetrievedChunk]]:
    """Retrieve relevant chunks from Pinecone. Returns (context_text, chunks)."""
    cfg = config or RAGConfig.from_env()
    if not cfg.rag_enabled:
        return "", []
    try:
        return retrieve_context(query, cfg)
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        return "", []


# ---------------------------------------------------------------------------
# 3. System Prompt Builder
# ---------------------------------------------------------------------------
def build_system_prompt(
    client_requirements_block: str,
    rag_context: str = "",
) -> str:
    """
    Assemble the full Arabic system prompt.
    Mirrors main.py section 6 — the exact same prompt template.
    """
    return f"""
أنت محرك تحليل هندسي ممتثل لكود البناء السعودي (SBC 1101) ومنصة بلدي والمنظومة التنظيمية للأكواد العمرانية للمدن السعودية.
مهمتك: مطابقة معطيات أرض العميل وتفضيلاته (JSON) مع نصوص الاشتراطات المسترجعة من قاعدة البيانات (Pinecone)، وإخراج تقرير فني موجز ومباشر.

⚠️ قوانين صارمة للأداء (Strict Constraints):
1. الرد يجب أن يكون مختصراً، مكثفاً بالبيانات، وخالياً تماماً من أي مقدمات ترحيبية، أو تذييل، أو عبارات إنشائية (مثل: "بصفتي مهندس"، "يسعدني تقديم").
2. ابدأ بكتابة التقرير فوراً مستخدماً النقاط (Bullet Points) والجداول لسهولة القراءة البرمجية.
3. إذا وجدت تعارضاً نظامياً (مثل بناء سكني على أرض تجارية)، اذكره في السطر الأول كـ "تحذير حرج".
4. اعتمد في حساباتك على الأرقام المسترجعة من Pinecone واعتبرها المرجعية العليا.
5. استخرج اسم (المدينة / المنطقة) بدقة من بيانات العميل، وقم بمطابقة متطلبات الواجهات والشكل الخارجي مع "الكود العمراني والدليل الإرشادي الخاص بتلك المدينة" (مثل الكود العمراني للمنطقة الشرقية، الميثاق العمراني السلماني بالرياض، إلخ) المتوفر في السياق.

الاشتراطات النظامية المسترجعة (Pinecone RAG Context):
--------------------------------------------------
{rag_context}
--------------------------------------------------

بيانات أرض ومعطيات العميل الحالية (Parsed JSON Data):
--------------------------------------------------
{client_requirements_block}
--------------------------------------------------

أخرج التقرير الهندسي النهائي مستخدماً هذا الهيكل الصارم فقط وبدون أي مقدمات:

### 1. ملخص الامتثال والجدوى النظامية
- حالة الاستخدام والتصنيف: [ممتثل / يوجد تعارض مع ذكر السبب في سطر واحد]
- طبيعة وموقع الأرض: أرض مفتوحة على 3 شوارع وممر (شبه بلوك مستقل) تمنح واجهات مفتوحة.
- أبعاد وأطوال أضلاع الأرض الحقيقية: شمالاً: [X]م، جنوباً: [X]م، شرقاً: [X]م، غرباً: [X]م.
- نسبة البناء المتاحة في هذه المنطقة: [النسبة من Pinecone] وتساوي [المساحة بالمتر المربع بناءً على مساحة الأرض]
- الارتدادات النظامية الصارمة المطلوبة: الأمامي (الجهة الغربية شارع 41م): [X]م، الجانبي والخلفي: [X]م.

### 2. محاور المداخل والحركة الرأسية (Circulation & Access)
- المداخل الرئيسية للمبنى: [حدد: مدخل ضيوف مستقل على الواجهة الغربية، مدخل عائلي جانبي مستقل، مدخل خدمات للمطبخ، كراج سيارات]
- عناصر الحركة الرأسية: [حدد: موقع درج الصالة الرئيسي وبئر المصعد في مركز صالة التوزيع لخدمة كبار السن بسهولة]

### 3. البرنامج المعماري وتوزيع الفراغات (Space Program & Zoning)
- كفاية المساحة للطلبات: [كافية / غير كافية مع تعليق رقمي مختصر]
- تفاصيل مكونات الأدوار (بالأعداد والخصائص):
  * الدور الأرضي: [الفناء الداخلي المفتوح "الحوش"، مجلس رجال بمدخل مستقل، صالة عائلية واسعة مطلة على الفناء، مطبخ مغلق مع مستودع ومطبخ تحضيري، جناح كبار السن بمواصفات خاصة، غرفة عاملة منزلية بدورة مياه]
  * الدور الأول: [جناح نوم رئيسي "ماستر" بغرفة ملابس، عدد (X) غرف نوم فرعية بدورات مياه، صالة عائلية علوية]
  * الملحق: [صالة متعددة الاستخدامات، منطقة غسيل وتخزين، سطح مفتوح بجلسة]

### 4. الطراز المعماري والهوية البصرية للشكل الخارجي (Urban & Facade Code)
- المدينة والمنطقة المستهدفة: [حدد اسم المدينة والمنطقة المستخرجة من الـ JSON]
- الطراز المعماري الإلزامي للمدينة: [تحديد الطراز بناءً على اشتراطات كود المدينة، مثل: المودرن المتوافق مع الهوية العمرانية للمنطقة الشرقية والدمام، أو الطراز السلماني للرياض، إلخ]
- محددات الواجهات والمواد الخارجية: [اذكر القيود الرقمية والنصية للألوان المسموحة، نسب فتحات النوافذ والزجاج، ومواد التشطيب الخارجية المفروضة حسب بلدية المدينة مثل الحجر، البروفايل، والتكسيات]
"""


# ---------------------------------------------------------------------------
# 4. Gemini Generation — Synchronous Full Response
# ---------------------------------------------------------------------------
def analyze_full(
    prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Call Gemini and return the full response text in one shot.
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not resolved_key:
        raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEY in .env.")

    config = RAGConfig.from_env()
    resolved_model = model or config.gemini_model

    client = genai.Client(api_key=resolved_key)
    response = client.models.generate_content(
        model=resolved_model,
        contents=prompt,
    )
    return response.text or ""


# ---------------------------------------------------------------------------
# 5. Gemini Generation — Async Streaming Generator
# ---------------------------------------------------------------------------
async def analyze_stream(
    prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields Gemini response chunks as they arrive.
    Used by the SSE endpoint to stream tokens to the frontend.
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not resolved_key:
        raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEY in .env.")

    config = RAGConfig.from_env()
    resolved_model = model or config.gemini_model

    # google-genai's sync streaming API, yielded asynchronously
    client = genai.Client(api_key=resolved_key)
    stream = client.models.generate_content_stream(
        model=resolved_model,
        contents=prompt,
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


# ---------------------------------------------------------------------------
# 6. Full Pipeline Orchestration
# ---------------------------------------------------------------------------
def run_analysis_pipeline(
    raw_input: str,
    default_land_area: float = 400.0,
    default_street_width: float = 15.0,
    default_style: str = "مودرن حديث",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[str, ParsedClientData, str, list[RetrievedChunk]]:
    """
    Execute the full analysis pipeline synchronously:
    1. Parse input
    2. Retrieve RAG context
    3. Build prompt
    4. Call Gemini (full response)

    Returns (report_text, parsed_data, rag_context, rag_chunks)
    """
    # Step 1: Parse
    client_data = parse_client_input(raw_input, default_land_area, default_street_width, default_style)

    # Step 2: RAG
    rag_context, rag_chunks = retrieve_rag_context(client_data.rag_query)

    # Step 3: Prompt
    prompt = build_system_prompt(client_data.client_requirements_block, rag_context)

    # Step 4: Gemini
    report = analyze_full(prompt, api_key=api_key, model=model)

    return report, client_data, rag_context, rag_chunks
