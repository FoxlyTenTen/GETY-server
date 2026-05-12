import json
import os
import re
import tempfile

import google.generativeai as genai
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from supabase_client import get_supabase, get_vector_store

# ── Disease section detection ─────────────────────────────────────────────────
_DISEASE_PATTERNS = [
    (r'\b(oidium|powdery.?mildew|luruhan daun sekunder oidium)\b',          'Powdery Mildew (Oidium)'),
    (r'\b(colletotrichum)\b',                                                'Colletotrichum'),
    (r'\b(corynespora)\b',                                                   'Corynespora'),
    (r'\b(fusicoccum|leaf.?blight|luruhan daun fusicoccum)\b',               'Fusicoccum Leaf Blight'),
    (r'\b(phytophthora palmivora|phytophthora abnormal)\b',                  'Phytophthora'),
    (r'\b(bird.?s?.?eye|bipolaris|rintik mata burung)\b',                    'Bird Eye Spot'),
    (r'\b(pink.?disease|corticium|cendawan angin)\b',                        'Pink Disease'),
    (r'\b(black.?stripe|calar hitam|phytophthora botryosa)\b',               'Black Stripe'),
    (r'\b(white.?root|akar putih|rigidoporus)\b',                            'White Root Disease'),
    (r'\b(red.?root|akar merah|ganoderma)\b',                                'Red Root Disease'),
    (r'\b(brown.?root|akar perang|phellinus)\b',                             'Brown Root Disease'),
]


def _detect_diseases(text: str) -> str:
    text_lower = text.lower()
    found = []
    for pattern, disease in _DISEASE_PATTERNS:
        if re.search(pattern, text_lower) and disease not in found:
            found.append(disease)
    return " | ".join(found) if found else "General"


def _extract_text_with_gemini(file_bytes: bytes, mime_type: str) -> str:
    """Upload PDF to Gemini Files API and extract all content."""
    if mime_type != "application/pdf":
        return file_bytes.decode("utf-8", errors="replace").strip()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        uploaded = genai.upload_file(tmp_path, mime_type="application/pdf")

        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={"temperature": 0},
        )
        response = model.generate_content([
            uploaded,
            (
                "Extract ALL content from this PDF about rubber tree (Hevea brasiliensis) diseases. "
                "Include every page: text, tables, and any text in figures or images.\n\n"
                "Format rules:\n"
                "1. Output text in correct reading order.\n"
                "2. Convert tables to plain text — one row per line, "
                "label fields clearly (e.g. 'Fungicide: Mancozeb | Rate: 0.2% | Interval: 2 weeks').\n"
                "3. Preserve ALL: disease names, fungicide names, chemical names, dosages, "
                "application rates, symptoms, treatment instructions, Malay terms.\n"
                "4. Remove page headers, footers, and page numbers.\n"
                "5. Output ONLY the extracted text. No commentary, no markdown."
            ),
        ])

        genai.delete_file(uploaded.name)
        return response.text.strip()
    finally:
        os.unlink(tmp_path)


def _restructure_with_gemini(raw_text: str) -> list[dict]:
    """
    Ask Gemini to restructure raw extracted text into a JSON array,
    one object per disease, with clearly labelled sections.
    Prints the result to terminal for inspection.
    """
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )

    prompt = f"""You are processing a rubber tree disease guide.
Restructure the text below into a JSON array. Each element represents ONE disease.

Return ONLY a JSON array with this exact structure for each disease:
[
  {{
    "disease_name": "exact disease name",
    "cause": "causal organism and how it spreads",
    "symptoms": "detailed description of all visible symptoms on leaves, bark, roots",
    "treatment_steps": "step by step treatment actions in order",
    "fungicides": "all recommended fungicide names",
    "dosage_and_rate": "exact dosage, concentration, mixing ratio, application rate for each fungicide",
    "spray_interval": "how often to apply and when to stop",
    "recovery_period": "estimated recovery timeline",
    "additional_notes": "any other important information"
  }}
]

Rules:
- One object per disease — never combine two diseases into one object
- Copy information EXACTLY as written — do not summarise or shorten
- If a field has no information in the text, use an empty string ""
- Include ALL diseases found in the text

TEXT:
{raw_text}"""

    response = model.generate_content(prompt)
    diseases = json.loads(response.text)

    # ── Print to terminal for inspection ──────────────────────────────────────
    print("\n" + "="*60)
    print("STRUCTURED DISEASE DATA (from Gemini)")
    print("="*60)
    print(json.dumps(diseases, indent=2, ensure_ascii=False))
    print("="*60 + "\n")

    return diseases


def _diseases_to_chunks(diseases: list[dict], file_id: str, filename: str) -> list[Document]:
    """
    Convert structured disease JSON into chunks.
    Each field (symptoms, treatment, etc.) becomes its own chunk,
    clearly labelled with disease name and section — no ambiguity.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = []
    chunk_index = 0

    for disease in diseases:
        name = disease.get("disease_name", "Unknown")

        # Each section becomes its own labelled text block
        sections = {
            "cause":            disease.get("cause", ""),
            "symptoms":         disease.get("symptoms", ""),
            "treatment_steps":  disease.get("treatment_steps", ""),
            "fungicides":       disease.get("fungicides", ""),
            "dosage_and_rate":  disease.get("dosage_and_rate", ""),
            "spray_interval":   disease.get("spray_interval", ""),
            "recovery_period":  disease.get("recovery_period", ""),
            "additional_notes": disease.get("additional_notes", ""),
        }

        for section, content in sections.items():
            if not content.strip():
                continue

            # Prefix every chunk with disease + section so the LLM always knows context
            labelled = f"Disease: {name}\nSection: {section}\n\n{content}"
            sub_chunks = splitter.create_documents(
                [labelled],
                metadatas=[{
                    "file_id": file_id,
                    "filename": filename,
                    "disease_name": name,
                    "section": section,
                    "chunk_index": chunk_index,
                }],
            )
            # Ensure chunk_index is unique across all chunks
            for doc in sub_chunks:
                doc.metadata["chunk_index"] = chunk_index
                chunk_index += 1

            chunks.extend(sub_chunks)

    return chunks


def process_file(file_id: str) -> dict:
    supabase = get_supabase()
    vector_store = get_vector_store()

    file_row = (
        supabase.table("knowledge_base_files")
        .select("*")
        .eq("id", file_id)
        .single()
        .execute()
        .data
    )
    if not file_row:
        raise ValueError(f"File not found: {file_id}")

    supabase.table("knowledge_base_files").update(
        {"processing_status": "processing", "processing_error": None}
    ).eq("id", file_id).execute()

    try:
        file_bytes = supabase.storage.from_("knowledge-base").download(
            file_row["file_path"]
        )

        # Step 1: Extract raw text from PDF via Gemini Files API
        print("\n[ingest] Step 1: Extracting text from PDF...")
        raw_text = _extract_text_with_gemini(file_bytes, file_row["mime_type"])
        if not raw_text:
            raise ValueError("No text extracted from file")
        print(f"[ingest] Extracted {len(raw_text)} characters")

        # Step 2: Restructure into per-disease JSON (prints to terminal)
        print("[ingest] Step 2: Restructuring into disease sections...")
        diseases = _restructure_with_gemini(raw_text)
        print(f"[ingest] Found {len(diseases)} diseases")

        # Step 3: Convert to labelled chunks
        print("[ingest] Step 3: Creating labelled chunks...")
        chunks = _diseases_to_chunks(diseases, file_id, file_row["filename"])
        if not chunks:
            raise ValueError("No chunks produced from file content")
        print(f"[ingest] Created {len(chunks)} chunks")

        # Step 4: Delete old chunks and store new ones
        supabase.table("documents").delete().filter(
            "metadata->>file_id", "eq", file_id
        ).execute()

        vector_store.add_documents(chunks)
        print(f"[ingest] Done — {len(chunks)} chunks stored in Supabase")

        supabase.table("knowledge_base_files").update(
            {
                "processing_status": "completed",
                "processing_error": None,
                "chunk_count": len(chunks),
            }
        ).eq("id", file_id).execute()

        return {"success": True, "chunks_created": len(chunks)}

    except Exception as err:
        supabase.table("knowledge_base_files").update(
            {"processing_status": "failed", "processing_error": str(err)}
        ).eq("id", file_id).execute()
        raise
