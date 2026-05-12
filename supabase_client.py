from functools import lru_cache

from supabase import Client, create_client
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore

import config


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    return create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=768,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> SupabaseVectorStore:
    return SupabaseVectorStore(
        client=get_supabase(),
        embedding=get_embeddings(),
        table_name="documents",
        query_name="match_documents",
    )


def hybrid_search(query: str, k: int = 8) -> list[Document]:
    """Combines semantic + keyword search via Supabase RPC (Reciprocal Rank Fusion)."""
    embedding = get_embeddings().embed_query(query)
    result = get_supabase().rpc("hybrid_search_documents", {
        "query_text": query,
        "query_embedding": embedding,
        "match_count": k,
    }).execute()

    docs = []
    for row in result.data or []:
        metadata = row.get("metadata") or {}
        metadata["similarity"] = row.get("similarity", 0)
        docs.append(Document(page_content=row["content"], metadata=metadata))
    return docs
