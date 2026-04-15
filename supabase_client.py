from functools import lru_cache

from supabase import Client, create_client
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
