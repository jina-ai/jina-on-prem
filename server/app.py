"""
Jina AI Air-Gapped Inference Server
Multi-schema embedding API: OpenAI, Voyage AI, Google Gemini, Cohere.
Real tok/s throughput measurement.

Endpoints:
  POST /v1/embeddings                          - OpenAI + Voyage AI compatible
  POST /v1/embed                               - Cohere compatible
  POST /v1/models/{model_id}:embedContent      - Gemini single-content
  POST /v1/models/{model_id}:batchEmbedContents - Gemini batch
  GET  /health                                 - health + throughput stats
"""

import os
import json
import time
import logging
import threading
from typing import Any, Optional, Union

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Path as FPath
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config from env ---
MODEL_ID = os.environ.get("JINA_MODEL_ID", "")
if os.environ.get("JINA_OFFLINE", "0") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Detect device
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    DEVICE = "cpu"
elif torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
logger.info(f"Using device: {DEVICE}")

app = FastAPI(
    title="Jina AI Air-Gapped Server",
    version="3.0.0",
    description="Multi-schema embedding server: OpenAI, Voyage AI, Gemini, Cohere",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Global model holder ---
MODEL = None
TOKENIZER = None
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


def _update_stats(n_tokens: int, elapsed_s: float) -> float:
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
    if TOKENIZER is not None:
        try:
            enc = TOKENIZER(texts, add_special_tokens=True)
            return sum(len(ids) for ids in enc["input_ids"])
        except Exception:
            pass
    return sum(len(t.split()) for t in texts)


def _embed(inputs: list[str], task: str = "retrieval", dimensions: Optional[int] = None):
    """Core embedding logic. Returns (embeddings_np, n_tokens, tok_per_s)."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    n_tokens = _count_tokens(inputs)
    t0 = time.perf_counter()
    with torch.no_grad():
        embeddings = MODEL.encode(
            inputs,
            task=task,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    elapsed = time.perf_counter() - t0

    tok_per_s = _update_stats(n_tokens, elapsed)
    logger.info(f"Embedded {len(inputs)} texts | {n_tokens} tokens | {elapsed*1000:.0f}ms | {tok_per_s:.0f} tok/s")

    if dimensions and dimensions < embeddings.shape[-1]:
        embeddings = embeddings[..., :dimensions]
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        embeddings = embeddings / np.where(norms > 0, norms, 1.0)

    return embeddings, n_tokens, tok_per_s


def load_model():
    global MODEL, TOKENIZER, MODEL_INFO

    model_id = MODEL_ID
    if not model_id:
        raise RuntimeError("JINA_MODEL_ID not set")

    logger.info(f"Loading model: {model_id}")

    from sentence_transformers import SentenceTransformer
    MODEL = SentenceTransformer(model_id, trust_remote_code=True, device=DEVICE)
    MODEL_INFO = {"model": model_id, "type": "embedding"}

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


# =============================================================================
# Health
# =============================================================================

@app.get("/health")
async def health():
    with _stats_lock:
        snap = dict(_stats)

    avg_tok_per_s = (
        snap["total_tokens"] / snap["total_latency_s"]
        if snap["total_latency_s"] > 0
        else None
    )

    resp = {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "ready": MODEL is not None,
        "schemas": ["openai", "voyage", "gemini", "cohere"],
    }

    if snap["total_requests"] > 0:
        resp["throughput"] = {
            "total_requests": snap["total_requests"],
            "total_tokens": snap["total_tokens"],
            "last_tok_per_s": round(snap["last_tok_per_s"], 1),
            "avg_tok_per_s": round(avg_tok_per_s, 1) if avg_tok_per_s else None,
            "peak_tok_per_s": round(snap["peak_tok_per_s"], 1),
        }

    return resp


# =============================================================================
# Schema 1: OpenAI + Voyage AI  (POST /v1/embeddings)
# Both use the same endpoint. Voyage adds input_type, output_dimension, output_dtype
# which map naturally to OpenAI's task, dimensions, encoding_format.
# =============================================================================

class OpenAIEmbeddingRequest(BaseModel):
    input: Union[str, list]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    task: Optional[str] = "retrieval"

    # Voyage AI extensions (aliases for OpenAI fields)
    input_type: Optional[str] = None        # Voyage: "query", "document", etc.
    output_dimension: Optional[int] = None  # Voyage alias for dimensions
    output_dtype: Optional[str] = None      # Voyage alias for encoding_format


class OpenAIEmbeddingResponse(BaseModel):
    object: str = "list"
    data: list
    model: str
    usage: dict


@app.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse, tags=["OpenAI / Voyage AI"])
async def create_embeddings_openai(request: OpenAIEmbeddingRequest):
    """
    OpenAI-compatible embedding endpoint. Also accepts Voyage AI fields:
    input_type (maps to task), output_dimension (maps to dimensions), output_dtype.

    Compatible with:
    - OpenAI client: openai.embeddings.create(model=..., input=...)
    - Voyage AI client: vo.embed(texts, model=..., input_type="query")
    - Elasticsearch inference service type: openai
    """
    inputs = request.input if isinstance(request.input, list) else [request.input]

    # Voyage AI field mapping
    dims = request.dimensions or request.output_dimension
    task = request.task or "retrieval"
    if request.input_type:
        # Map Voyage input_type to Jina task
        voyage_task_map = {
            "query": "retrieval.query",
            "document": "retrieval.passage",
            "classification": "classification",
            "clustering": "clustering",
        }
        task = voyage_task_map.get(request.input_type, request.input_type)

    embeddings, n_tokens, tok_per_s = _embed(inputs, task=task, dimensions=dims)

    data = [
        {"object": "embedding", "embedding": emb.tolist(), "index": i}
        for i, emb in enumerate(embeddings)
    ]

    return OpenAIEmbeddingResponse(
        data=data,
        model=MODEL_ID,
        usage={
            "prompt_tokens": n_tokens,
            "total_tokens": n_tokens,
            "tok_per_s": round(tok_per_s, 1),
        },
    )


# =============================================================================
# Schema 2: Cohere  (POST /v1/embed)
# =============================================================================

class CohereEmbedRequest(BaseModel):
    texts: list[str]
    model: Optional[str] = None
    input_type: Optional[str] = "search_document"  # search_query, search_document, classification, clustering
    truncate: Optional[str] = "END"
    embedding_types: Optional[list[str]] = Field(default_factory=lambda: ["float"])


class CohereEmbedResponse(BaseModel):
    id: str
    texts: list[str]
    embeddings: dict
    meta: dict
    response_type: str = "embeddings_floats"


@app.post("/v1/embed", tags=["Cohere"])
async def create_embeddings_cohere(request: CohereEmbedRequest):
    """
    Cohere-compatible embedding endpoint.

    Compatible with:
    - cohere.Client().embed(texts=[...], model=..., input_type="search_query")
    """
    # Map Cohere input_type to Jina task
    cohere_task_map = {
        "search_query": "retrieval.query",
        "search_document": "retrieval.passage",
        "classification": "classification",
        "clustering": "clustering",
    }
    task = cohere_task_map.get(request.input_type or "search_document", "retrieval")

    embeddings, n_tokens, tok_per_s = _embed(request.texts, task=task)

    embedding_list = [emb.tolist() for emb in embeddings]

    import uuid
    return {
        "id": str(uuid.uuid4()),
        "texts": request.texts,
        "embeddings": {
            "float": embedding_list,
        },
        "meta": {
            "api_version": {"version": "1"},
            "billed_units": {"input_tokens": n_tokens},
            "tok_per_s": round(tok_per_s, 1),
        },
        "response_type": "embeddings_floats",
    }


# =============================================================================
# Schema 3: Google Gemini  (POST /v1/models/{model}:embedContent)
# =============================================================================

class GeminiPart(BaseModel):
    text: str


class GeminiContent(BaseModel):
    parts: list[GeminiPart]
    role: Optional[str] = None


class GeminiEmbedContentRequest(BaseModel):
    content: GeminiContent
    taskType: Optional[str] = None  # RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, etc.
    title: Optional[str] = None
    outputDimensionality: Optional[int] = None


class GeminiBatchEmbedRequest(BaseModel):
    requests: list[dict]  # list of GeminiEmbedContentRequest-like dicts


@app.post("/v1/models/{model_id}:embedContent", tags=["Google Gemini"])
async def embed_content_gemini(
    model_id: str,
    request: GeminiEmbedContentRequest,
):
    """
    Google Gemini-compatible single embedding endpoint.

    Compatible with:
    - genai.embed_content(model=..., content=..., task_type="RETRIEVAL_QUERY")
    - POST /v1/models/jina-embeddings-v5-text-nano:embedContent
    """
    text = " ".join(p.text for p in request.content.parts)

    gemini_task_map = {
        "RETRIEVAL_QUERY": "retrieval.query",
        "RETRIEVAL_DOCUMENT": "retrieval.passage",
        "SEMANTIC_SIMILARITY": "text-matching",
        "CLASSIFICATION": "classification",
        "CLUSTERING": "clustering",
        "QUESTION_ANSWERING": "retrieval.query",
        "FACT_VERIFICATION": "retrieval.query",
        "CODE_RETRIEVAL_QUERY": "retrieval.query",
    }
    task = gemini_task_map.get(request.taskType or "", "retrieval")

    embeddings, n_tokens, tok_per_s = _embed([text], task=task, dimensions=request.outputDimensionality)

    return {
        "embedding": {
            "values": embeddings[0].tolist(),
        },
        "metadata": {
            "tokenCount": n_tokens,
            "tok_per_s": round(tok_per_s, 1),
        },
    }


@app.post("/v1/models/{model_id}:batchEmbedContents", tags=["Google Gemini"])
async def batch_embed_contents_gemini(
    model_id: str,
    request: GeminiBatchEmbedRequest,
):
    """
    Google Gemini-compatible batch embedding endpoint.

    Compatible with:
    - genai.batch_embed_contents(requests=[...])
    - POST /v1/models/jina-embeddings-v5-text-nano:batchEmbedContents
    """
    texts = []
    dims = None

    for req in request.requests:
        content = req.get("content", {})
        parts = content.get("parts", [])
        text = " ".join(p.get("text", "") for p in parts)
        texts.append(text)
        if not dims and req.get("outputDimensionality"):
            dims = req["outputDimensionality"]

    # Use task from first request
    first_task_type = request.requests[0].get("taskType", "") if request.requests else ""
    gemini_task_map = {
        "RETRIEVAL_QUERY": "retrieval.query",
        "RETRIEVAL_DOCUMENT": "retrieval.passage",
        "SEMANTIC_SIMILARITY": "text-matching",
        "CLASSIFICATION": "classification",
        "CLUSTERING": "clustering",
    }
    task = gemini_task_map.get(first_task_type, "retrieval")

    embeddings, n_tokens, tok_per_s = _embed(texts, task=task, dimensions=dims)

    return {
        "embeddings": [{"values": emb.tolist()} for emb in embeddings],
        "metadata": {
            "tokenCount": n_tokens,
            "tok_per_s": round(tok_per_s, 1),
        },
    }


# =============================================================================
# Reranker endpoint (bonus - for reranker models)
# =============================================================================

class RerankRequest(BaseModel):
    query: str
    documents: list[Union[str, dict]]
    model: Optional[str] = None
    top_n: Optional[int] = None
    return_documents: Optional[bool] = True


@app.post("/v1/rerank", tags=["Reranker"])
async def rerank(request: RerankRequest):
    """OpenAI-style rerank endpoint (for reranker models)."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    docs = []
    for d in request.documents:
        docs.append(d if isinstance(d, str) else d.get("text", ""))

    t0 = time.perf_counter()
    with torch.no_grad():
        # sentence-transformers CrossEncoder interface
        pairs = [[request.query, doc] for doc in docs]
        try:
            scores = MODEL.predict(pairs)
        except AttributeError:
            raise HTTPException(status_code=400, detail="Loaded model does not support reranking")
    elapsed = time.perf_counter() - t0

    results = [
        {"index": i, "relevance_score": float(s), "document": {"text": docs[i]} if request.return_documents else None}
        for i, s in enumerate(scores)
    ]
    results.sort(key=lambda x: x["relevance_score"], reverse=True)

    if request.top_n:
        results = results[:request.top_n]

    return {
        "model": MODEL_ID,
        "results": results,
        "meta": {"elapsed_ms": round(elapsed * 1000, 1)},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
