import os
import re
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai

from rag.config import RAGConfig
from rag.retriever import retrieve_context
from services.floor_plan_service import generate_floor_plan

# تحميل ملف الإعدادات البيئية
load_dotenv(Path(__file__).resolve().parent / ".env")
config = RAGConfig.from_env()

# إعدادات الصفحة في Streamlit
st.set_page_config(page_title="مساعدك المعماري الذكي", layout="centered")
st.title(" 🏗️ المساعد المعماري السعودي التجريبي")
st.write("أدخل تفاصيل أرضك واحتياجات عائلتك، أو قم بلصق ملف الـ JSON الخاص بالصك، وسيقوم النظام بفحصها وفق كود البناء السعودي.")

# =========================================================
# 1. إعدادات الشريط الجانبي (Sidebar)
# =========================================================
with st.sidebar:
    st.header("بيانات الأرض والنظام")
    land_area = st.number_input("مساحة الأرض الافتراضية (متر مربع)", min_value=100, value=400)
    street_width = st.number_input("عرض الشارع الافتراضي (متر)", min_value=5, value=15)
    style = st.selectbox("الطراز المعماري المفضل", ["طراز سلماني", "مودرن حديث", "نجدي مطور"])

    st.divider()
    st.header("إعدادات Gemini")
    default_api_key = os.getenv("GEMINI_API_KEY", "")
    try:
        default_api_key = default_api_key or st.secrets["GEMINI_API_KEY"]
    except (FileNotFoundError, KeyError):
        pass
    api_key = st.text_input(
        "Gemini API Key",
        value=default_api_key,
        type="password",
        help="احصل على المفتاح من Google AI Studio: https://aistudio.google.com/apikey",
    )

    st.divider()
    st.header("إعدادات Wan2.7 (DashScope)")
    default_dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")
    try:
        default_dashscope_key = default_dashscope_key or st.secrets["DASHSCOPE_API_KEY"]
    except (FileNotFoundError, KeyError):
        pass
    dashscope_api_key = st.text_input(
        "DashScope API Key",
        value=default_dashscope_key,
        type="password",
        help="مفتاح Alibaba DashScope. احصل عليه من: https://dashscope.console.aliyun.com/",
    )
    dashscope_region = st.radio(
        "المنطقة / Region",
        options=["🌍 خارج الصين (International)", "🇨🇳 داخل الصين (China)"],
        index=0,
        horizontal=True,
        help="اختر International إذا كنت خارج الصين (السعودية، الإمارات...). هذا يحدد عنوان DashScope API.",
    )
    use_intl_endpoint = "صين" not in dashscope_region
    generate_floor_plan_enabled = st.toggle(
        "توليد مخطط الطابق 2D",
        value=bool(default_dashscope_key),
        disabled=not bool(dashscope_api_key),
        help="يتطلب مفتاح DashScope. يرسل تقرير المهندس إلى Wan2.7-image-pro لرسم المخطط.",
    )
    if dashscope_api_key:
        endpoint_label = "dashscope-intl.aliyuncs.com" if use_intl_endpoint else "dashscope.aliyuncs.com"
        st.caption(f"🔗 Endpoint: `{endpoint_label}`")

    st.divider()
    st.header("RAG / Pinecone")
    use_rag = st.toggle("تفعيل RAG", value=config.rag_enabled, disabled=not config.rag_enabled)
    if config.rag_enabled:
        st.caption(f"Index: `{config.pinecone_index}`")
        st.caption(f"Namespace: `{config.pinecone_namespace}`")
        st.caption("لتغذية قاعدة المعرفة: `python ingest.py`")
    else:
        st.warning("أضف `PINECONE_API_KEY` و `PINECONE_INDEX` في `.env` لتفعيل RAG.")

# =========================================================
# 2. إدارة وعرض صندوق المحادثة (Chat UI)
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# استقبال مدخلات المستخدم (نص عادي أو JSON صريح)
if user_query := st.chat_input("اكتب تفاصيل عائلتك أو الصق نص الـ JSON الكامل هنا..."):
    
    # [تعديل جوهري]: نجهز المتغيرات التمهيدية قبل العرض
    current_land_area = land_area
    current_street_width = street_width
    current_style = style

    is_json_input = False
    parsed_json = {}
    
    try:
        parsed_json = json.loads(user_query)
        is_json_input = True
    except json.JSONDecodeError:
        is_json_input = False

    # دالة فرعية لاستخراج عرض الشارع رقمياً من النصوص
    def parse_street_width(boundary_data):
        desc = boundary_data.get("description", "")
        length = boundary_data.get("length_m", 0.0)
        match = re.search(r'عرض\s*(\d+(?:\.\d+)?)', desc)
        width = float(match.group(1)) if match else 0.0
        return {"length": length, "street_width": width, "desc": desc}

    # =========================================================
    # 3. معالجة وتفكيك بيانات الـ JSON (Parsing Stage) - تم تقديمها هنا
    # =========================================================
    if is_json_input:
        # أ) تفاصيل الوثيقة والصك
        doc_data = parsed_json.get("document", {})
        doc_number = doc_data.get("document_number", "")
        doc_type = doc_data.get("document_type", "")
        operation_type = doc_data.get("operation_type", "")
        
        owners = parsed_json.get("owners", [])
        owner_names = [owner.get("name", "") for owner in owners]

        # ب) معلومات العقار الجغرافية والتصنيف النطاقي
        property_info = parsed_json.get("property", {})
        city = property_info.get("city", "")
        district = property_info.get("district", "")
        plan_number = property_info.get("plan_number", "")
        parcel_number = property_info.get("parcel_number", "")
        
        # تحديث مساحة الأرض فورياً من الـ JSON
        current_land_area = property_info.get("area_m2", current_land_area)

        # ج) هندسة حدود الأرض وعروض الشوارع الأربعة
        boundaries = parsed_json.get("land_details", {}).get("boundaries", {})
        parsed_boundaries = {
            "north": parse_street_width(boundaries.get("north", {})),
            "south": parse_street_width(boundaries.get("south", {})),
            "east": parse_street_width(boundaries.get("east", {})),
            "west": parse_street_width(boundaries.get("west", {}))
        }

        # حساب الشارع الأكبر لتحديد الارتدادات والارتفاعات القصوى نظاماً
        all_widths = [b["street_width"] for b in parsed_boundaries.values()]
        if any(all_widths):
            current_street_width = max(all_widths)

        # د) تفاصيل العائلة ومصفوفة الاحتياجات الداخلية
        family_prefs = parsed_json.get("family_preferences", {})
        estimated_family_count = family_prefs.get("family_members", {}).get("estimated_count", 4)
        has_elderly = family_prefs.get("elderly_or_accessibility", {}).get("has_elderly", False)
        
        bedrooms_count = family_prefs.get("bedrooms", {}).get("count", 3)
        master_count = family_prefs.get("master_bedrooms", {}).get("count", 1)
        guest_reception_label = family_prefs.get("guest_reception", {}).get("label", "")
        kitchen_label = family_prefs.get("kitchen", {}).get("label", "")
        
        additional_rooms = family_prefs.get("additional_rooms", {})
        has_maid = additional_rooms.get("maid_room", False)
        has_driver = additional_rooms.get("driver_room", False)
        has_laundry = additional_rooms.get("laundry_room", False)
        has_storage = additional_rooms.get("storage_room", False)
        
        restrictions = parsed_json.get("extra_information", {}).get("restrictions", "لا يوجد")

        # صياغة النص النظيف والمفكك برمجياً ليعرض للمستخدم وللـ Prompt
        client_requirements_block = f"""
### 📋 البيانات المستخرجة من الصك والطلب بنجاح:
* **وثيقة العقار:** {doc_type} رقم `{doc_number}` ({operation_type})
* **الموقع:** مدينة {city}، حي {district} (مخطط رقم: {plan_number})
* **التصنيف النطاقي:** {parcel_number}
* **مساحة الأرض المعتمدة:** {current_land_area} متر مربع
* **حدود وعروض الشوارع المحيطة للأرض:**
    * **شمالاً:** بطول {parsed_boundaries['north']['length']}م ({parsed_boundaries['north']['desc']})
    * **جنوباً:** بطول {parsed_boundaries['south']['length']}م ({parsed_boundaries['south']['desc']})
    * **شرقاً:** بطول {parsed_boundaries['east']['length']}م ({parsed_boundaries['east']['desc']})
    * **غرباً:** بطول {parsed_boundaries['west']['length']}م ({parsed_boundaries['west']['desc']})
* **تفاصيل العائلة والاحتياجات الممررة:**
    * عدد الأفراد: {estimated_family_count} أشخاص | كبار سن/احتياجات خاصة: {"نعم" if has_elderly else "لا"}
    * غرف النوم: {bedrooms_count} (منها غرف ماستر: {master_count})
    * نمط الاستقبال: {guest_reception_label} | نمط المطبخ: {kitchen_label}
    * غرف خدمية إضافية: عاملة منزليّة ({"نعم" if has_maid else "لا"})، سائق ({"نعم" if has_driver else "لا"})، غسيل ({"نعم" if has_laundry else "لا"})، مستودع ({"نعم" if has_storage else "لا"})
* **قيود إضافية:** {restrictions}
"""
        rag_query = f"اشتراطات كود البناء أمانة مدينة {city} حي {district} تصنيف {parcel_number} شوارع عرض {[w for w in all_widths if w > 0]}"
    else:
        client_requirements_block = f'طلب المستخدم المباشر: "{user_query}"'
        rag_query = user_query

    # [تعديل العرض]: الآن نعرض ونحفظ النص المفكك النظيف بدلاً من الـ JSON الخام في الشات
    with st.chat_message("user"):
        st.markdown(client_requirements_block)
    st.session_state.messages.append({"role": "user", "content": client_requirements_block})

    # =========================================================
    # 4. استدعاء نظام الـ RAG (Pinecone Integration)
    # =========================================================
    rag_context = ""
    retrieved_chunks = []
    if use_rag and config.rag_enabled:
        with st.spinner("جاري البحث في قاعدة المعرفة المخصصة لكود البناء..."):
            try:
                rag_context, retrieved_chunks = retrieve_context(rag_query, config)
            except Exception as exc:
                st.warning(f"تعذر استرجاع السياق من Pinecone: {exc}")

    rag_section = ""
    if rag_context:
        rag_section = f"""
    ## مراجع مسترجعة من قاعدة المعرفة (Pinecone)
    استخدم المعلومات التالية كمصدر قانوني أساسي عند صياغة الإجابة والالتزام بالأرقام المذكورة فيها:
    {rag_context}
    """

    # =========================================================
    # 6. هندسة البرومبت وصياغة الـ System Instructions
    # =========================================================
    system_prompt = f"""
أنت محرك تحليل هندسي ممتثل لكود البناء السعودي (SBC 1101) ومنصة بلدي. 
مهمتك: مطابقة معطيات أرض العميل وتفضيلاته (JSON) مع نصوص الاشتراطات المسترجعة من قاعدة البيانات (Pinecone)، وإخراج تقرير فني موجز ومباشر.

⚠️ قوانين صارمة للأداء (Strict Constraints):
1. الرد يجب أن يكون مختصراً، مكثفاً بالبيانات، وخالياً تماماً من أي مقدمات ترحيبية، أو تذييل، أو عبارات إنشائية (مثل: "بصفتي مهندس"، "يسعدني تقديم").
2. ابدأ بكتابة التقرير فوراً مستخدماً النقاط (Bullet Points) والجداول لسهولة القراءة البرمجية.
3. إذا وجدت تعارضاً نظامياً (مثل بناء سكني على أرض تجارية)، ا ذكره في السطر الأول كـ "تحذير حرج".
4. اعتمد في حساباتك على الأرقام المسترجعة من Pinecone واعتبرها المرجعية العليا.

الاشتراطات النظامية المسترجعة (Pinecone RAG Context):
--------------------------------------------------
{rag_context}
--------------------------------------------------

بيانات أرض ومعطيات العميل الحالية (Parsed JSON Data):
--------------------------------------------------
{client_requirements_block}
--------------------------------------------------

أخرج التقرير الهندسي النهائي مستخدمماً هذا الهيكل الصارم فقط وبدون أي مقدمات:

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
    """

    # =========================================================
    # 7. استدعاء نموذج جيميناي وبث الإجابة (Gemini Streaming)
    # =========================================================
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if retrieved_chunks:
            with st.expander("📚 المصادر المرجعية المسترجعة من Pinecone"):
                for chunk in retrieved_chunks:
                    st.markdown(f"**{chunk.source}** — نسبة المطابقة: `{chunk.score:.3f}`")
                    st.write(chunk.text)

        if not api_key:
            full_response = (
                "⚠️ **مفتاح Gemini API مطلوب لتقديم الاستشارة.**\n\n"
                "أدخل المفتاح في الشريط الجانبي، أو قم بوضعه in ملف `.env` تحت متغير `GEMINI_API_KEY`."
            )
            message_placeholder.markdown(full_response)
        else:
            try:
                client = genai.Client(api_key=api_key)
                stream = client.models.generate_content_stream(
                    model=config.gemini_model,
                    contents=system_prompt,
                )
                for chunk in stream:
                    if chunk.text:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as exc:
                full_response = f"❌ **حدث خطأ أثناء الاتصال بنماذج Gemini:**\n\n`{exc}`"
                message_placeholder.markdown(full_response)

        # =========================================================
        # 8. توليد مخطط الطابق الثنائي الأبعاد (Wan2.7-image-pro)
        # =========================================================
        if full_response and dashscope_api_key and generate_floor_plan_enabled:
            with st.expander("🏠 عرض مخطط الطابق الثنائي الأبعاد (2D Floor Plan)", expanded=False):
                with st.spinner("جاري إرسال التقرير إلى Wan2.7-image-pro لتوليد المخطط المعماري..."):
                    try:
                        floor_plan_bytes = generate_floor_plan(
                            report_text=full_response,
                            api_key=dashscope_api_key,
                            use_intl=use_intl_endpoint,
                        )
                        if floor_plan_bytes:
                            st.image(
                                floor_plan_bytes,
                                caption="مخطط ثنائي الأبعاد مُولَّد بواسطة Wan2.7-image-pro (DashScope)",
                                use_container_width=True,
                            )
                            st.download_button(
                                label="⬇️ تحميل المخطط (PNG)",
                                data=floor_plan_bytes,
                                file_name="floor_plan.png",
                                mime="image/png",
                            )
                        else:
                            st.warning("⚠️ لم يتم توليد صورة من Wan2.7-image-pro. تحقق من صلاحيات مفتاح DashScope.")
                    except Exception as fp_exc:
                        st.error(f"❌ خطأ في خدمة توليد المخطط (Wan2.7): `{fp_exc}`")

    st.session_state.messages.append({"role": "assistant", "content": full_response})