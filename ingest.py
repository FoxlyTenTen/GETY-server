from io import BytesIO

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from supabase_client import get_supabase, get_vector_store


def _extract_text(file_bytes: bytes, mime_type: str) -> str:
    if mime_type == "application/pdf":
        reader = PdfReader(BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
    if mime_type == "text/plain":
        return file_bytes.decode("utf-8", errors="replace").strip()
    raise ValueError(f"Unsupported mime_type: {mime_type}")


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

        text = _extract_text(file_bytes, file_row["mime_type"])
        if not text:
            raise ValueError("No text extracted from file")

        # Delete any existing chunks for this file (re-processing)
        supabase.table("documents").delete().filter(
            "metadata->>file_id", "eq", file_id
        ).execute()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        chunks = splitter.create_documents(
            [text],
            metadatas=[
                {
                    "file_id": file_id,
                    "filename": file_row["filename"],
                }
            ],
        )
        # Stamp chunk_index into metadata for citation
        for idx, doc in enumerate(chunks):
            doc.metadata["chunk_index"] = idx

        if not chunks:
            raise ValueError("No chunks produced from file content")

        vector_store.add_documents(chunks)

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
