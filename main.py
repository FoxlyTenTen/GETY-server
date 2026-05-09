from typing import Literal
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import io

import config
from ingest import process_file
from agent import query_rag

# ── TFLite model setup ────────────────────────────────────────────
_MODEL_PATH = Path(__file__).parent.parent / "model" / "best_float32.tflite"
_IMG_SIZE = 224
_CLASSES = ['Bird_Eye_Spot', 'Colletotrichum', 'Corynespora',
            'Healthy', 'Leaf_Blight', 'Powdery_Mildew']

_interpreter = None
_input_details = None
_output_details = None


def _get_interpreter():
    global _interpreter, _input_details, _output_details
    if _interpreter is None:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
        _interpreter = Interpreter(model_path=str(_MODEL_PATH))
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()[0]
        _output_details = _interpreter.get_output_details()[0]
    return _interpreter, _input_details, _output_details


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


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Run TFLite inference on an uploaded leaf image."""
    if not _MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Model not found at {_MODEL_PATH}")
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize((_IMG_SIZE, _IMG_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0

        interp, in_det, out_det = _get_interpreter()

        if in_det['shape'][1] == 3:  # CHW layout
            arr = arr.transpose(2, 0, 1)
        arr = np.expand_dims(arr, axis=0).astype(in_det['dtype'])

        interp.set_tensor(in_det['index'], arr)
        interp.invoke()
        probs = interp.get_tensor(out_det['index'])[0]

        top_idx = int(np.argmax(probs))
        return {
            "disease": _CLASSES[top_idx],
            "confidence": float(probs[top_idx]),
            "all_probabilities": {_CLASSES[i]: float(probs[i]) for i in range(len(_CLASSES))},
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
