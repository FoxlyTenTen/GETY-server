# GETY RAG Backend

Agentic RAG backend for the GETY mobile app. FastAPI + LangChain + Gemini + Supabase pgvector.

## What it does

- `POST /process-file` — downloads an uploaded file from Supabase Storage, extracts text, chunks it, generates embeddings via Gemini, stores them in the `documents` table.
- `POST /query` — tool-calling agent that retrieves relevant chunks via pgvector similarity search and generates an answer with source citations.
- `GET /health` — liveness check.

## One-time setup

### 1. Supabase SQL

Run this in the Supabase SQL Editor (Dashboard → SQL Editor → New Query):

```sql
-- Drop old chunks infra (from previous edge function approach)
DROP FUNCTION IF EXISTS match_chunks(vector, int, float);
DROP TABLE IF EXISTS knowledge_base_chunks;

-- LangChain's SupabaseVectorStore expects UUID ids (it generates them client-side)
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT,
    metadata JSONB,
    embedding VECTOR(768)
);

CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- LangChain's expected RPC signature (id must be uuid to match the table)
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(768),
    match_count int DEFAULT NULL,
    filter jsonb DEFAULT '{}'
) RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    similarity float
) LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT documents.id, documents.content, documents.metadata,
           1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE documents.metadata @> filter
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- RLS: only the backend (service_role) touches this
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_all" ON documents FOR ALL USING (auth.role() = 'service_role');
```

### 2. Get Supabase service role key

Supabase Dashboard → **Settings** → **API** → under **Project API keys** copy the **`service_role`** secret (the long one — NOT the anon key). This key bypasses RLS. Never commit it.

### 3. Create `.env`

```powershell
copy .env.example .env
```

Then edit `.env` and fill in:
- `GOOGLE_API_KEY` — your Gemini key
- `SUPABASE_SERVICE_ROLE_KEY` — the key from step 2

`SUPABASE_URL` is already set to the GETY project.

### 4. Install dependencies

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run locally

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs for the interactive FastAPI docs.

### Letting your phone reach the backend

`localhost` only works on the laptop itself — from a phone on the same WiFi, you need the laptop's LAN IP:

1. Run `ipconfig` in PowerShell
2. Under your active WiFi adapter, find `IPv4 Address` (e.g. `192.168.1.42`)
3. In the GETY mobile app's `.env` file, set:
   ```
   EXPO_PUBLIC_RAG_BACKEND_URL=http://192.168.1.42:8000
   ```
4. Restart Expo (`npx expo start`) so the new env var is picked up
5. If the phone still can't connect, Windows Defender Firewall may be blocking port 8000. Allow inbound TCP 8000 for `python.exe` / `uvicorn.exe`.

## Smoke tests

```powershell
# Health check
curl http://localhost:8000/health

# Query with empty KB
curl -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d '{\"question\":\"what is powdery mildew\",\"chat_history\":[]}'
```

## Deploy (later)

Render / Railway / Fly.io all run `uvicorn` out of the box. Set the same env vars in the platform dashboard, then update `EXPO_PUBLIC_RAG_BACKEND_URL` in the GETY app `.env` to the deployed HTTPS URL and rebuild the app (Expo env vars are baked at build time).
