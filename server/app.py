"""
Jina AI Air-Gapped Inference Server
OpenAI-compatible embedding API with real tok/s throughput measurement.
"""

import os
import json
import time
import logging
import threading
from typing import Optional, Union

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config from env ---
MODEL_ID = os.environ.get("JINA_MODEL_ID", "")
# Enforce offline mode only if explicitly set (e.g., inside Docker with baked-in weights)
if os.environ.get("JINA_OFFLINE", "0") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Detect device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
logger.info(f"Using device: {DEVICE}")

app = FastAPI(title="Jina AI Air-Gapped Server", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Global model holder ---
MODEL = None
TOKENIZER = None  # for real token counting
MODEL_INFO = {}

# --- Throughput stats (thread-safe) ---
_stats_lock = threading.Lock()
_stats = {
    "total_requests": 0,
    "total_tokens": 0,
    "total_latency_s": 0.0,
    "last_tok_per_s": 0.0,
    "peak_tok_per_s": 0.0,
}


def _update_stats(n_tokens: int, elapsed_s: float):
    tok_per_s = n_tokens / elapsed_s if elapsed_s > 0 else 0.0
    with _stats_lock:
        _stats["total_requests"] += 1
        _stats["total_tokens"] += n_tokens
        _stats["total_latency_s"] += elapsed_s
        _stats["last_tok_per_s"] = tok_per_s
        if tok_per_s > _stats["peak_tok_per_s"]:
            _stats["peak_tok_per_s"] = tok_per_s
    return tok_per_s


def _count_tokens(texts: list[str]) -> int:
    """Count tokens using the actual tokenizer."""
    if TOKENIZER is not None:
        try:
            enc = TOKENIZER(texts, add_special_tokens=True)
            return sum(len(ids) for ids in enc["input_ids"])
        except Exception:
            pass
    # fallback: whitespace split approximation
    return sum(len(t.split()) for t in texts)


def load_model():
    global MODEL, TOKENIZER, MODEL_INFO

    model_id = MODEL_ID
    if not model_id:
        raise RuntimeError("JINA_MODEL_ID not set")

    logger.info(f"Loading model: {model_id}")

    from sentence_transformers import SentenceTransformer
    MODEL = SentenceTransformer(model_id, trust_remote_code=True, device=DEVICE)
    MODEL_INFO = {"model": model_id, "type": "embedding"}

    # Load tokenizer for real token counting
    try:
        from transformers import AutoTokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        logger.info("Tokenizer loaded for real tok/s measurement")
    except Exception as e:
        logger.warning(f"Could not load tokenizer ({e}), will use word-split approximation")

    logger.info(f"Model loaded: {model_id} on {DEVICE}")


@app.on_event("startup")
async def startup():
    load_model()


# --- Request/Response schemas ---

class EmbeddingRequest(BaseModel):
    input: Union[str, list]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    task: Optional[str] = "retrieval"


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list
    model: str
    usage: dict


# --- Endpoints ---

@app.get("/health")
async def health():
    with _stats_lock:
        stats_snapshot = dict(_stats)

    avg_tok_per_s = (
        stats_snapshot["total_tokens"] / stats_snapshot["total_latency_s"]
        if stats_snapshot["total_latency_s"] > 0
        else None
    )

    resp = {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "ready": MODEL is not None,
    }

    if stats_snapshot["total_requests"] > 0:
        resp["throughput"] = {
            "total_requests": stats_snapshot["total_requests"],
            "total_tokens": stats_snapshot["total_tokens"],
            "last_tok_per_s": round(stats_snapshot["last_tok_per_s"], 1),
            "avg_tok_per_s": round(avg_tok_per_s, 1) if avg_tok_per_s else None,
            "peak_tok_per_s": round(stats_snapshot["peak_tok_per_s"], 1),
        }

    return resp


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = request.input if isinstance(request.input, list) else [request.input]

    # Count real tokens before encoding
    n_tokens = _count_tokens(inputs)

    t0 = time.perf_counter()
    with torch.no_grad():
        embeddings = MODEL.encode(
            inputs,
            task=request.task or "retrieval",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    elapsed = time.perf_counter() - t0

    tok_per_s = _update_stats(n_tokens, elapsed)
    logger.info(
        f"Embedded {len(inputs)} texts | {n_tokens} tokens | "
        f"{elapsed*1000:.0f}ms | {tok_per_s:.0f} tok/s"
    )

    # Handle Matryoshka truncation
    if request.dimensions and request.dimensions < embeddings.shape[-1]:
        embeddings = embeddings[..., :request.dimensions]
        embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)

    data = [
        {"object": "embedding", "embedding": emb.tolist(), "index": i}
        for i, emb in enumerate(embeddings)
    ]

    return EmbeddingResponse(
        data=data,
        model=MODEL_ID,
        usage={
            "prompt_tokens": n_tokens,
            "total_tokens": n_tokens,
            "tok_per_s": round(tok_per_s, 1),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
