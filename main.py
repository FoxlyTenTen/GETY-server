from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from ingest import process_file
from agent import query_rag


app = FastAPI(title="GETY RAG Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessFileRequest(BaseModel):
    file_id: str = Field(..., min_length=1)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chat_history: list[ChatMessage] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process-file")
def process_file_endpoint(body: ProcessFileRequest):
    try:
        return process_file(body.file_id)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.post("/query")
def query_endpoint(body: QueryRequest):
    try:
        return query_rag(
            body.question,
            [msg.model_dump() for msg in body.chat_history],
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
