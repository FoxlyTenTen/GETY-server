import json

from langchain_google_genai import ChatGoogleGenerativeAI

from supabase_client import get_vector_store, hybrid_search

# ── Targeted search queries per disease class ─────────────────────────────────
_DISEASE_SEARCH_MAP = {
    "Bird_Eye_Spot":  "Bird Eye Spot bipolaris heveae rintik mata burung propineb mancozeb",
    "Colletotrichum": "Colletotrichum gloeosporioides luruhan daun sekunder chlorothalonil propineb",
    "Corynespora":    "Corynespora cassiicola luruhan daun tulang ikan benomyl bernlate",
    "Leaf_Blight":    "Fusicoccum leaf blight luruhan daun fusicoccum carbendazim propiconazole",
    "Powdery_Mildew": "Oidium heveae powdery mildew luruhan daun sekunder sulphur tridemorph",
}

_DISPLAY_NAME_MAP = {
    "Bird_Eye_Spot":  "Bird Eye Spot",
    "Colletotrichum": "Colletotrichum (Anthracnose)",
    "Corynespora":    "Corynespora Leaf Fall",
    "Leaf_Blight":    "Fusicoccum Leaf Blight",
    "Powdery_Mildew": "Powdery Mildew (Oidium)",
}

# Maps display name keywords → what to filter by in metadata
_DISEASE_METADATA_NAMES = {
    "Bird_Eye_Spot":  "Bird Eye Spot",
    "Colletotrichum": "Colletotrichum",
    "Corynespora":    "Corynespora",
    "Leaf_Blight":    "Fusicoccum Leaf Blight",
    "Powdery_Mildew": "Powdery Mildew (Oidium)",
}


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        generation_config={"response_mime_type": "application/json"},
    )


def _retrieve_context(disease_class: str, k: int = 15) -> str:
    """
    Retrieve chunks using a targeted disease-specific query.
    Chunks are labelled with their disease_name metadata so the LLM
    can identify and discard off-topic chunks.
    Prioritises chunks tagged with the correct disease.
    """
    query = _DISEASE_SEARCH_MAP.get(disease_class, disease_class.replace("_", " "))
    target_disease = _DISEASE_METADATA_NAMES.get(disease_class, "")

    try:
        docs = hybrid_search(query, k=k)
    except Exception:
        docs = get_vector_store().similarity_search(query, k=k)
    if not docs:
        return "No relevant documents found in knowledge base."

    # Sort: chunks tagged with the correct disease come first (handles multi-disease tags)
    def sort_key(doc):
        chunk_disease = doc.metadata.get("disease_name", "General")
        return 0 if target_disease in chunk_disease else 1

    docs.sort(key=sort_key)

    # Label each chunk so the LLM knows which disease it belongs to
    return "\n\n---\n\n".join(
        f"[Disease tag: {doc.metadata.get('disease_name', 'General')}]\n{doc.page_content}"
        for doc in docs
    )


def get_disease_info(disease_class: str) -> dict:
    display_name = _DISPLAY_NAME_MAP.get(disease_class, disease_class.replace("_", " "))
    context = _retrieve_context(disease_class)

    prompt = f"""You are an expert in rubber tree (Hevea brasiliensis) disease management.

CRITICAL: You must answer ONLY about the disease: {display_name}
The knowledge base below may contain information about multiple diseases.
Use ONLY chunks tagged with [Disease tag: {_DISEASE_METADATA_NAMES.get(disease_class, display_name)}] or chunks that explicitly mention {display_name}.
Completely ignore any chunks about other diseases.

KNOWLEDGE BASE:
{context}

Return a single JSON object with EXACTLY these fields (no extra fields, no markdown):
{{
  "disease_name": "{display_name}",
  "risk_level": "Low" or "Medium" or "High",
  "description": "2-3 sentence practical description of symptoms and impact of {display_name} only",
  "what_to_do": ["action 1 specific to {display_name}", "action 2", "action 3"],
  "prevention_tips": [
    {{"title": "short title", "desc": "one sentence specific to {display_name}"}},
    {{"title": "short title", "desc": "one sentence specific to {display_name}"}}
  ],
  "recommended_fungicide": "fungicide name used for {display_name}",
  "water_mix_ratio": "e.g. 0.2% (a.i/L)",
  "estimated_recovery_days": <integer>,
  "follow_up_days": <integer>
}}

Rules:
- Exactly 2 prevention_tips, exactly 3 what_to_do items
- risk_level must be exactly one of: Low, Medium, High
- Every field must describe {display_name} specifically — never another disease
- No markdown, no extra fields"""

    llm = _get_llm()
    response = llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)
    result = json.loads(content)
    result["disease_name"] = display_name
    return result


def generate_milestones(disease_class: str, recovery_days: int) -> dict:
    display_name = _DISPLAY_NAME_MAP.get(disease_class, disease_class.replace("_", " "))
    context = _retrieve_context(disease_class, k=6)

    mid1 = round(recovery_days * 0.3)
    mid2 = round(recovery_days * 0.65)

    prompt = f"""You are an expert in rubber tree (Hevea brasiliensis) disease management.

CRITICAL: Create a milestone plan ONLY for the disease: {display_name}
The knowledge base below may contain chunks about multiple diseases.
Use ONLY chunks tagged with [Disease tag: {_DISEASE_METADATA_NAMES.get(disease_class, display_name)}] or chunks that explicitly mention {display_name}.
Completely ignore chunks about other diseases.

KNOWLEDGE BASE:
{context}

Create a 4-step treatment milestone plan for {display_name}. Recovery period: {recovery_days} days.

Return a single JSON object (no extra fields, no markdown):
{{
  "steps": [
    {{"title": "step title", "description": "one actionable sentence for {display_name}", "day_offset": 0}},
    {{"title": "step title", "description": "one actionable sentence for {display_name}", "day_offset": {mid1}}},
    {{"title": "step title", "description": "one actionable sentence for {display_name}", "day_offset": {mid2}}},
    {{"title": "step title", "description": "one actionable sentence for {display_name}", "day_offset": {recovery_days}}}
  ],
  "expert_tip": "one practical sentence specific to {display_name} only"
}}

Rules:
- Exactly 4 steps, day_offset values must be 0, {mid1}, {mid2}, {recovery_days} in order
- Every step and expert_tip must be specific to {display_name} — never mention another disease
- No markdown, no extra fields"""

    llm = _get_llm()
    response = llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)
    return json.loads(content)
