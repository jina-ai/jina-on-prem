"""
Jina AI Air-Gapped Inference Server
OpenAI-compatible API for embedding, reranking, and reading.
"""

import os
import json
import time
import logging
from typing import Optional, Union

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config from env ---
MODEL_ID = os.environ.get("JINA_MODEL_ID", "")
MODEL_TYPE = os.environ.get("JINA_MODEL_TYPE", "embedding")  # embedding | reranker | reader
# Enforce offline mode only if explicitly set (e.g., inside Docker with baked-in weights)
# When JINA_OFFLINE=1 is set, lock down all network access
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

app = FastAPI(title="Jina AI Air-Gapped Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Global model holder ---
MODEL = None
TOKENIZER = None
MODEL_INFO = {}


def load_model():
    global MODEL, TOKENIZER, MODEL_INFO

    model_id = MODEL_ID
    if not model_id:
        raise RuntimeError("JINA_MODEL_ID not set")

    logger.info(f"Loading model: {model_id} (type={MODEL_TYPE})")

    if MODEL_TYPE == "embedding":
        from sentence_transformers import SentenceTransformer
        MODEL = SentenceTransformer(model_id, trust_remote_code=True, device=DEVICE)
        MODEL_INFO = {"model": model_id, "type": "embedding"}

    elif MODEL_TYPE == "reranker":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        MODEL = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE)
        MODEL.eval()
        MODEL_INFO = {"model": model_id, "type": "reranker"}

    elif MODEL_TYPE == "reader":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
        )
        if DEVICE != "cuda":
            MODEL = MODEL.to(DEVICE)
        MODEL.eval()
        MODEL_INFO = {"model": model_id, "type": "reader"}

    logger.info(f"Model loaded: {model_id}")


@app.on_event("startup")
async def startup():
    load_model()


# --- Request/Response schemas ---

class EmbeddingRequest(BaseModel):
    input: Union[str, list]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    task: Optional[str] = "retrieval.query"


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: list
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list
    model: str
    usage: dict


class RerankRequest(BaseModel):
    model: Optional[str] = None
    query: str
    documents: list
    top_n: Optional[int] = None
    return_documents: Optional[bool] = True


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[dict] = None


class RerankResponse(BaseModel):
    model: str
    results: list
    usage: dict


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list
    usage: dict


# --- Endpoints ---

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "type": MODEL_TYPE,
        "device": DEVICE,
        "ready": MODEL is not None,
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    if MODEL is None or MODEL_TYPE != "embedding":
        raise HTTPException(status_code=503, detail="Embedding model not loaded")

    inputs = request.input if isinstance(request.input, list) else [request.input]

    # Encode
    with torch.no_grad():
        embeddings = MODEL.encode(
            inputs,
            task=request.task or "retrieval.query",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    # Handle Matryoshka truncation
    if request.dimensions and request.dimensions < embeddings.shape[-1]:
        embeddings = embeddings[..., :request.dimensions]
        embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)

    data = [
        {"object": "embedding", "embedding": emb.tolist(), "index": i}
        for i, emb in enumerate(embeddings)
    ]

    total_tokens = sum(len(s.split()) for s in inputs)  # approx

    return EmbeddingResponse(
        data=data,
        model=MODEL_ID,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    )


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    if MODEL is None or MODEL_TYPE != "reranker":
        raise HTTPException(status_code=503, detail="Reranker model not loaded")

    query = request.query
    docs = request.documents

    # Normalize docs to strings
    doc_texts = []
    for d in docs:
        if isinstance(d, str):
            doc_texts.append(d)
        elif isinstance(d, dict):
            doc_texts.append(d.get("text", str(d)))
        else:
            doc_texts.append(str(d))

    pairs = [[query, doc] for doc in doc_texts]

    with torch.no_grad():
        inputs = TOKENIZER(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)
        scores = MODEL(**inputs).logits.squeeze(-1).float().cpu().numpy()

    if len(scores.shape) > 1:
        scores = scores[:, 0]

    # Sort by score descending
    ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
    top_n = request.top_n or len(ranked)

    results = []
    for rank_idx, (orig_idx, score) in enumerate(ranked[:top_n]):
        result = {"index": orig_idx, "relevance_score": float(score)}
        if request.return_documents:
            d = docs[orig_idx]
            result["document"] = d if isinstance(d, dict) else {"text": d}
        results.append(result)

    total_tokens = len(query.split()) + sum(len(t.split()) for t in doc_texts)

    return RerankResponse(
        model=MODEL_ID,
        results=results,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if MODEL is None or MODEL_TYPE != "reader":
        raise HTTPException(status_code=503, detail="Reader model not loaded")

    # Build prompt from messages
    prompt = TOKENIZER.apply_chat_template(
        [{"role": m["role"], "content": m["content"]} for m in request.messages],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=request.max_tokens or 2048,
            do_sample=request.temperature > 0,
            temperature=request.temperature if request.temperature > 0 else 1.0,
            pad_token_id=TOKENIZER.eos_token_id,
        )

    # Decode only new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = TOKENIZER.decode(new_tokens, skip_special_tokens=True)

    prompt_tokens = inputs["input_ids"].shape[1]
    completion_tokens = len(new_tokens)

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# Also support /rerank (some clients use this)
@app.post("/rerank")
async def rerank_alias(request: RerankRequest):
    return await rerank(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
