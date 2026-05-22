"""
Jina AI Air-Gapped Inference Server
Multi-schema embedding API: OpenAI, Voyage AI, Google Gemini, Cohere.
Real tok/s throughput measurement.

Endpoints:
  POST /v1/embeddings                          - OpenAI + Voyage AI compatible (text + multimodal)
  POST /v1/embed                               - Cohere compatible (text + multimodal)
  POST /v1/multimodalembeddings                - Voyage AI multimodal endpoint
  POST /v1/models/{model_id}:embedContent      - Gemini single-content (text + multimodal)
  POST /v1/models/{model_id}:batchEmbedContents - Gemini batch
  GET  /health                                 - health + throughput stats
"""

import base64
import io
import os
import json
import sys
import time
import logging
import threading
from pathlib import Path
from typing import Any, Optional, Union

# Register dynamic model module directories so transformers check_imports
# can find sibling modules (e.g. configuration_eurobert imported by
# modeling_jina_embeddings_v5). Without this, air-gapped containers fail
# because importlib.util.find_spec cannot locate these modules.
_hf_modules = Path(os.environ.get("HF_HOME", "")) / "modules" / "transformers_modules"
if _hf_modules.is_dir():
    for _commit_dir in _hf_modules.rglob("*.py"):
        _parent = str(_commit_dir.parent)
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
    del _commit_dir, _parent
del _hf_modules

import torch
import numpy as np

# Patch transformers for air-gapped Jina model loading.
# 1. resolve_trust_remote_code: force True so custom_st.py AutoConfig/AutoModel
#    calls don't prompt for interactive confirmation in offline containers.
# 2. add_generation_mixin_to_remote_model: guard against models without
#    prepare_inputs_for_generation (embedding-only models like JinaEmbeddingsV5).
try:
    from transformers import dynamic_module_utils as _dmu
    _orig_resolve = _dmu.resolve_trust_remote_code
    def _always_trust(*args, **kwargs):
        return True
    _dmu.resolve_trust_remote_code = _always_trust
except Exception:
    pass
try:
    from transformers.models.auto import auto_factory as _af
    _orig_add_gen = getattr(_af, 'add_generation_mixin_to_remote_model', None)
    if _orig_add_gen:
        def _safe_add_gen(model_class):
            if not hasattr(model_class, 'prepare_inputs_for_generation'):
                return model_class
            return _orig_add_gen(model_class)
        _af.add_generation_mixin_to_remote_model = _safe_add_gen
except Exception:
    pass

# Boost matmul precision: uses TF32 on Ampere/Ada/Hopper GPUs, ~1.2x faster on L4
torch.set_float32_matmul_precision('high')
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

# --- Runtime optimizations (applied at load time) ---
# CPU: use all physical cores (0 = auto-detect from env or os.cpu_count)
_omp_env = os.environ.get("OMP_NUM_THREADS", "")
_n_cpu_threads = int(_omp_env) if _omp_env and _omp_env != "0" else (os.cpu_count() or 4)
torch.set_num_threads(_n_cpu_threads)
torch.set_num_interop_threads(max(1, _n_cpu_threads // 2))

# GPU: enable cuDNN auto-tuner and TF32 matmul (no effect on CPU/MPS)
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

app = FastAPI(
    title="Jina AI Air-Gapped Server",
    version="4.0.0",
    description="Multi-schema embedding server: OpenAI, Voyage AI, Gemini, Cohere (text + multimodal)",
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

# --- Multimodal config ---
# Models that support image / audio / video inputs
MULTIMODAL_MODEL_IDS = {
    "jina-embeddings-v5-omni-small",
    "jina-embeddings-v5-omni-nano",
    "jina-embeddings-v4",
    "jina-clip-v2",
    "jina-clip-v1",
    "jina-reranker-m0",
    "jina-vlm",
}

MAX_MEDIA_BYTES = 10 * 1024 * 1024  # 10 MB per input

IMAGE_MIMES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif",
    "image/bmp", "image/tiff", "image/avif", "image/heic", "image/svg+xml",
}
AUDIO_MIMES = {
    "audio/wav", "audio/x-wav", "audio/mp3", "audio/mpeg", "audio/flac",
    "audio/ogg", "audio/m4a", "audio/x-m4a", "audio/opus",
}
VIDEO_MIMES = {
    "video/mp4", "video/avi", "video/x-msvideo", "video/quicktime",
    "video/x-matroska", "video/webm", "video/x-flv", "video/x-ms-wmv",
}


# =============================================================================
# Multimodal helpers
# =============================================================================

def _is_omni_model() -> bool:
    """True if the loaded model supports multimodal (image/audio/video) inputs."""
    short_id = MODEL_ID.split("/")[-1] if MODEL_ID else ""
    return short_id in MULTIMODAL_MODEL_IDS


def _require_omni():
    """Raise HTTP 400 if the current model does not support multimodal inputs."""
    if not _is_omni_model():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{MODEL_ID}' is text-only and does not accept image/audio/video inputs. "
                f"Use one of: {sorted(MULTIMODAL_MODEL_IDS)}"
            ),
        )


def _decode_b64(b64_str: str) -> tuple:
    """
    Decode a base64 string in raw or data-URL format.
    Returns (raw_bytes, mime_type).

    Accepts:
    - Raw base64: "iVBORw0KGgo..."
    - Data URL:   "data:image/png;base64,iVBORw0KGgo..."
    """
    mime_type = ""
    data = b64_str.strip()
    if data.startswith("data:"):
        try:
            header, data = data.split(",", 1)
            mime_type = header.split(";")[0][5:]  # strip "data:"
        except ValueError:
            raise HTTPException(status_code=400, detail="Malformed data URL")
    try:
        raw = base64.b64decode(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")
    if len(raw) > MAX_MEDIA_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Media too large: {len(raw):,} bytes exceeds {MAX_MEDIA_BYTES:,} byte limit",
        )
    return raw, mime_type


def _detect_mime(raw: bytes, hint: str = "") -> str:
    """Detect MIME type from magic bytes; fall back to hint."""
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:2] == b"\xff\xd8":
        return "image/jpeg"
    if len(raw) >= 6 and raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "audio/wav"
    if len(raw) >= 3 and raw[:3] == b"ID3":
        return "audio/mp3"
    if len(raw) >= 4 and raw[:4] == b"fLaC":
        return "audio/flac"
    if len(raw) >= 12 and b"ftyp" in raw[4:12]:
        return "video/mp4"
    if len(raw) >= 4 and raw[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    return hint


def _bytes_to_st_input(raw: bytes, mime_hint: str = ""):
    """
    Convert raw bytes to the input type expected by sentence-transformers encode().
    - image/* -> PIL.Image.Image
    - audio/* or video/* -> io.BytesIO (sentence-transformers accepts bytes/BytesIO)
    """
    mime = (_detect_mime(raw, mime_hint) or mime_hint).lower()

    if mime in IMAGE_MIMES:
        from PIL import Image
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
            return img
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")

    if mime in AUDIO_MIMES or mime in VIDEO_MIMES:
        return io.BytesIO(raw)

    # Unknown MIME: attempt image decode, fall back to BytesIO
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img
    except Exception:
        return io.BytesIO(raw)


def _parse_typed_base64(item: dict, key: str, default_mime: str) -> list:
    """Parse image_base64 / audio_base64 / video_base64 typed fields."""
    inner = item.get(key, "")
    if isinstance(inner, str):
        raw, mime = _decode_b64(inner)
        mime = mime or default_mime
    elif isinstance(inner, dict):
        raw, mime = _decode_b64(inner.get("base64", inner.get("data", "")))
        mime = mime or inner.get("mime_type", inner.get("mimeType", default_mime))
    else:
        raise HTTPException(status_code=400, detail=f"Invalid value for '{key}'")
    return [_bytes_to_st_input(raw, mime)]


def _parse_content_part(part: dict) -> list:
    """
    Parse a single content-array part into a list of ST-compatible inputs.

    Handles:
    - {"type": "text", "text": "..."}
    - {"type": "image", "format": "base64", "value": "..."}         (Elastic format)
    - {"type": "image_url", "image_url": {"url": "data:..."}}       (Cohere/Voyage)
    - {"type": "image_base64", "image_base64": "data:..." | {...}}
    - {"type": "audio_base64", "audio_base64": "data:..." | {...}}
    - {"type": "video_base64", "video_base64": "data:..." | {...}}
    - {"inlineData": {"mimeType": "...", "data": "..."}}             (Gemini format)
    """
    if not isinstance(part, dict):
        raise HTTPException(
            status_code=400,
            detail=f"Content part must be a dict, got {type(part).__name__}",
        )

    t = part.get("type", "")

    if t == "text":
        return [part.get("text", "")]

    # Elastic Inference Service: {"type": "image", "format": "base64", "value": "..."}
    if t == "image" and part.get("format") == "base64":
        raw, mime = _decode_b64(part["value"])
        return [_bytes_to_st_input(raw, mime or "image/jpeg")]

    # Cohere/Voyage: {"type": "image_url", "image_url": {"url": "data:..."}}
    if t == "image_url":
        url_val = part.get("image_url", {})
        url = url_val.get("url", url_val) if isinstance(url_val, dict) else str(url_val)
        if not url.startswith("data:"):
            raise HTTPException(
                status_code=400,
                detail="image_url: only data: URLs (base64) are supported in air-gapped mode",
            )
        raw, mime = _decode_b64(url)
        return [_bytes_to_st_input(raw, mime)]

    # Gemini: {"inlineData": {"mimeType": "...", "data": "..."}}
    if "inlineData" in part:
        inline = part["inlineData"]
        raw, _ = _decode_b64(inline.get("data", ""))
        return [_bytes_to_st_input(raw, inline.get("mimeType", ""))]

    # Typed base64 formats
    if t == "image_base64" or "image_base64" in part:
        return _parse_typed_base64(part, "image_base64", "image/jpeg")
    if t == "audio_base64" or "audio_base64" in part:
        return _parse_typed_base64(part, "audio_base64", "audio/wav")
    if t == "video_base64" or "video_base64" in part:
        return _parse_typed_base64(part, "video_base64", "video/mp4")

    raise HTTPException(status_code=400, detail=f"Unknown content part type: '{t}'")


def _parse_openai_item(item) -> list:
    """
    Parse one element of the OpenAI `input` array into a list of ST-compatible inputs.

    Returns a list:
    - [str]          -> plain text (len 1)
    - [PIL.Image]    -> single image (len 1)
    - [BytesIO]      -> single audio/video (len 1)
    - [x, y, ...]    -> fused multimodal parts (len > 1, caller wraps in tuple)

    Supported formats:
    1. "plain text"
    2. {"type": "text", "text": "..."}
    3. {"type": "image", "format": "base64", "value": "..."}                    (Elastic)
    4. {"type": "image_base64", "image_base64": {"base64": "...", "mime_type": "..."}}
    5. {"type": "audio_base64", "audio_base64": {"base64": "...", "mime_type": "..."}}
    6. {"type": "video_base64", "video_base64": {"base64": "...", "mime_type": "..."}}
    7. {"content": [...]}  fused multimodal block -> parts merged into ONE embedding
    """
    if isinstance(item, str):
        return [item]

    if not isinstance(item, dict):
        raise HTTPException(
            status_code=400,
            detail=f"Input item must be str or dict, got {type(item).__name__}",
        )

    # Fused content block: {"content": [...]}
    if "content" in item and isinstance(item["content"], list):
        parts = []
        for p in item["content"]:
            parts.extend(_parse_content_part(p))
        return parts

    # Single-part item: delegate to content-part parser
    return _parse_content_part(item)


# =============================================================================
# Core embedding (text-only and multimodal)
# =============================================================================

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


def _count_tokens(texts: list) -> int:
    if TOKENIZER is not None:
        try:
            enc = TOKENIZER(texts, add_special_tokens=True)
            return sum(len(ids) for ids in enc["input_ids"])
        except Exception:
            pass
    return sum(len(t.split()) for t in texts)


def _resolve_task(task: str) -> str:
    """Adapt task name to what the loaded model actually supports.

    v5 models accept: retrieval, text-matching, classification, clustering.
    v3 models accept: retrieval.query, retrieval.passage, separation,
                      classification, text-matching.

    API endpoints use fine-grained names (retrieval.query, retrieval.passage)
    from schema mapping. This function adapts them per model.
    """
    short_id = MODEL_ID.split("/")[-1] if MODEL_ID else ""
    is_v3 = "v3" in short_id and "v5" not in short_id
    is_v5 = "v5" in short_id

    if is_v3:
        v3_map = {
            "retrieval": "retrieval.passage",
            "retrieval.query": "retrieval.query",
            "retrieval.passage": "retrieval.passage",
            "text-matching": "text-matching",
            "classification": "classification",
            "clustering": "text-matching",
            "separation": "separation",
        }
        return v3_map.get(task, "retrieval.passage")
    elif is_v5:
        v5_map = {
            "retrieval": "retrieval",
            "retrieval.query": "retrieval",
            "retrieval.passage": "retrieval",
            "text-matching": "text-matching",
            "classification": "classification",
            "clustering": "clustering",
        }
        return v5_map.get(task, "retrieval")
    else:
        # Older models: strip sub-task
        if task.startswith("retrieval"):
            return "retrieval"
        return task


def _embed(inputs: list, task: str = "retrieval", dimensions: Optional[int] = None):
    """Text-only embedding. Returns (embeddings_np, n_tokens, tok_per_s)."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    task = _resolve_task(task)
    n_tokens = _count_tokens(inputs)
    t0 = time.perf_counter()
    ctx = torch.autocast("cuda", dtype=torch.float16) if DEVICE == "cuda" else torch.inference_mode()
    with torch.inference_mode(), ctx:
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


def _embed_mixed(items: list, task: str = "retrieval", dimensions: Optional[int] = None):
    """
    Embed a mixed list of inputs (text strings, PIL.Images, BytesIO, or lists for fused multimodal).

    Each items[i] is:
    - str / PIL.Image / BytesIO  -> one embedding
    - list of the above         -> fused multimodal tuple -> one embedding

    Returns (embeddings_np, n_tokens, tok_per_s).
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    task = _resolve_task(task)

    encode_inputs = []
    text_parts = []

    for item in items:
        if isinstance(item, list):
            # Sort: put non-text items (PIL.Image, BytesIO) first in the tuple.
            # The model's _encode_single_image has a bug when called via the
            # shortcut at line 903 (text-first ordering). Keeping non-text first
            # forces _encode_composite_parts which handles all orderings correctly.
            sorted_parts = sorted(item, key=lambda x: isinstance(x, str))
            encode_inputs.append(tuple(sorted_parts))
            for p in item:
                if isinstance(p, str):
                    text_parts.append(p)
        elif isinstance(item, str):
            encode_inputs.append(item)
            text_parts.append(item)
        else:
            # Standalone PIL.Image or BytesIO: wrap with empty string to produce a
            # 2-element tuple (non-text first). This routes through _encode_composite_parts
            # and avoids the _encode_single_image processor bug.
            encode_inputs.append((item, ""))

    n_tokens = _count_tokens(text_parts) if text_parts else len(items)

    t0 = time.perf_counter()
    ctx = torch.autocast("cuda", dtype=torch.float16) if DEVICE == "cuda" else torch.inference_mode()
    with torch.inference_mode(), ctx:
        embeddings = MODEL.encode(
            encode_inputs,
            task=task,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    elapsed = time.perf_counter() - t0

    if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
        embeddings = embeddings[None, :]

    tok_per_s = _update_stats(n_tokens, elapsed)
    logger.info(
        f"Embedded {len(items)} multimodal inputs | {n_tokens} text-tokens | "
        f"{elapsed*1000:.0f}ms | {tok_per_s:.0f} tok/s"
    )

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

    # --- Apply dtype optimization (GPU only) ---
    if DEVICE == "cuda":
        dtype_env = os.environ.get("JINA_DTYPE", "float16").lower()
        if dtype_env in ("float16", "fp16", "half"):
            MODEL.half()
            logger.info("Model converted to FP16 (JINA_DTYPE=float16)")
        elif dtype_env in ("bfloat16", "bf16"):
            MODEL.bfloat16()
            logger.info("Model converted to BF16 (JINA_DTYPE=bfloat16)")
        else:
            logger.info(f"Running in FP32 (JINA_DTYPE={dtype_env})")

        # torch.compile: fuses ops, ~10-30% additional speedup
        try:
            first_module = MODEL._first_module()
            if hasattr(first_module, "auto_model"):
                first_module.auto_model = torch.compile(
                    first_module.auto_model,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                logger.info("torch.compile(reduce-overhead) applied to encoder")
        except Exception as e:
            logger.warning(f"torch.compile skipped: {e}")

    multimodal = _is_omni_model()
    logger.info(f"Model loaded: {model_id} on {DEVICE} | multimodal={multimodal} | threads={_n_cpu_threads}")


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
        "multimodal": _is_omni_model(),
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
#
# Multimodal: `input` items may be structured dicts in addition to plain strings.
# Formats:
#   {"type": "image", "format": "base64", "value": "<b64>"}            (Elastic)
#   {"type": "image_base64", "image_base64": {"base64":"...", "mime_type":"image/png"}}
#   {"type": "audio_base64", "audio_base64": {"base64":"...", "mime_type":"audio/wav"}}
#   {"type": "video_base64", "video_base64": {"base64":"...", "mime_type":"video/mp4"}}
#   {"content": [part, ...]}  fused multimodal -> ONE embedding per block
# =============================================================================

class OpenAIEmbeddingRequest(BaseModel):
    input: Union[str, list]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    task: Optional[str] = "retrieval"

    # Voyage AI extensions
    input_type: Optional[str] = None
    output_dimension: Optional[int] = None
    output_dtype: Optional[str] = None


class OpenAIEmbeddingResponse(BaseModel):
    object: str = "list"
    data: list
    model: str
    usage: dict


@app.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse, tags=["OpenAI / Voyage AI"])
async def create_embeddings_openai(request: OpenAIEmbeddingRequest):
    """
    OpenAI-compatible embedding endpoint. Also accepts Voyage AI fields.
    Supports multimodal inputs for omni models via structured input items.

    Compatible with:
    - OpenAI client: openai.embeddings.create(model=..., input=...)
    - Voyage AI client: vo.embed(texts, model=..., input_type="query")
    - Elasticsearch inference service type: openai
    """
    raw_inputs = request.input if isinstance(request.input, list) else [request.input]

    dims = request.dimensions or request.output_dimension
    task = request.task or "retrieval"
    if request.input_type:
        voyage_task_map = {
            "query": "retrieval.query",
            "document": "retrieval.passage",
            "classification": "classification",
            "clustering": "clustering",
        }
        task = voyage_task_map.get(request.input_type, "retrieval")

    has_multimodal = any(not isinstance(x, str) for x in raw_inputs)

    if has_multimodal:
        _require_omni()
        parsed = []
        for item in raw_inputs:
            parts = _parse_openai_item(item)
            parsed.append(parts if len(parts) > 1 else parts[0])
        embeddings, n_tokens, tok_per_s = _embed_mixed(parsed, task=task, dimensions=dims)
    else:
        embeddings, n_tokens, tok_per_s = _embed(raw_inputs, task=task, dimensions=dims)

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
#
# Multimodal extensions:
# - Legacy: {"texts": [...], "images": ["data:image/png;base64,..."]}
# - V2:     {"inputs": [{"content": [{"type":"image_url","image_url":{"url":"data:..."}}, ...]}]}
# =============================================================================

class CohereEmbedRequest(BaseModel):
    texts: Optional[list] = None
    images: Optional[list] = None       # legacy: list of data-URL strings
    inputs: Optional[list] = None       # V2: content-block array
    model: Optional[str] = None
    input_type: Optional[str] = "search_document"
    truncate: Optional[str] = "END"
    embedding_types: Optional[list] = Field(default_factory=lambda: ["float"])


@app.post("/v1/embed", tags=["Cohere"])
async def create_embeddings_cohere(request: CohereEmbedRequest):
    """
    Cohere-compatible embedding endpoint.

    Text-only:
      {"texts": [...], "input_type": "search_document"}

    Legacy multimodal:
      {"images": ["data:image/png;base64,..."], "input_type": "search_document"}

    V2 multimodal:
      {"inputs": [{"content": [{"type": "image_url", "image_url": {"url": "data:..."}}, {"type": "text", "text": "..."}]}]}
    """
    cohere_task_map = {
        "search_query": "retrieval.query",
        "search_document": "retrieval.passage",
        "classification": "classification",
        "clustering": "clustering",
    }
    task = cohere_task_map.get(request.input_type or "search_document", "retrieval")

    import uuid

    if request.inputs is not None:
        # V2 inputs format
        _require_omni()
        parsed = []
        for inp in request.inputs:
            content = inp.get("content", [])
            parts = []
            for p in content:
                parts.extend(_parse_content_part(p))
            parsed.append(parts if len(parts) > 1 else (parts[0] if parts else ""))
        embeddings, n_tokens, tok_per_s = _embed_mixed(parsed, task=task)
        texts_out = []
        images_count = len(parsed)
    elif request.images is not None:
        # Legacy: images as data-URL list
        _require_omni()
        parsed = []
        for img_data_url in request.images:
            raw, mime = _decode_b64(img_data_url)
            parsed.append(_bytes_to_st_input(raw, mime))
        embeddings, n_tokens, tok_per_s = _embed_mixed(parsed, task=task)
        texts_out = request.texts or []
        images_count = len(request.images)
    else:
        # Text-only
        texts = request.texts or []
        embeddings, n_tokens, tok_per_s = _embed(texts, task=task)
        texts_out = texts
        images_count = 0

    embedding_list = [emb.tolist() for emb in embeddings]

    return {
        "id": str(uuid.uuid4()),
        "texts": texts_out,
        "embeddings": {
            "float": embedding_list,
        },
        "meta": {
            "api_version": {"version": "2"},
            "billed_units": {
                "input_tokens": n_tokens,
                "images": images_count,
            },
            "tok_per_s": round(tok_per_s, 1),
        },
        "response_type": "embeddings_floats",
    }


# =============================================================================
# Schema 3: Voyage AI Multimodal  (POST /v1/multimodalembeddings)
#
# Format:
# {
#   "inputs": [
#     {"content": [
#       {"type": "text", "text": "..."},
#       {"type": "image_base64", "image_base64": "data:image/jpeg;base64,..."},
#       {"type": "video_base64", "video_base64": "data:video/mp4;base64,..."}
#     ]}
#   ],
#   "model": "voyage-multimodal-3.5",
#   "input_type": "document"
# }
# =============================================================================

class VoyageMultimodalRequest(BaseModel):
    inputs: list
    model: Optional[str] = None
    input_type: Optional[str] = None
    truncation: Optional[bool] = True


@app.post("/v1/multimodalembeddings", tags=["Voyage AI Multimodal"])
async def create_multimodal_embeddings_voyage(request: VoyageMultimodalRequest):
    """
    Voyage AI-compatible multimodal embedding endpoint.

    Each input is {"content": [text/image/video/audio parts]}.
    All parts within one input are fused into a single embedding.

    Compatible with the Voyage AI multimodalembeddings REST API.
    """
    _require_omni()

    voyage_task_map = {
        "query": "retrieval.query",
        "document": "retrieval.passage",
    }
    task = voyage_task_map.get(request.input_type or "", "retrieval")

    parsed = []
    for inp in request.inputs:
        content = inp.get("content", [])
        parts = []
        for p in content:
            parts.extend(_parse_content_part(p))
        parsed.append(parts if len(parts) > 1 else (parts[0] if parts else ""))

    embeddings, n_tokens, tok_per_s = _embed_mixed(parsed, task=task)

    return {
        "embeddings": [emb.tolist() for emb in embeddings],
        "text_tokens": n_tokens,
        "image_pixels": 0,  # not tracked
        "total_tokens": n_tokens,
        "model": MODEL_ID,
    }


# =============================================================================
# Schema 4: Google Gemini  (POST /v1/models/{model}:embedContent)
#
# Multimodal: parts may include inlineData in addition to text:
#   {"inlineData": {"mimeType": "image/png", "data": "base64..."}}
# =============================================================================

class GeminiPart(BaseModel):
    text: Optional[str] = None
    inlineData: Optional[dict] = None   # {"mimeType": "image/png", "data": "base64..."}


class GeminiContent(BaseModel):
    parts: list                          # list of GeminiPart-like dicts
    role: Optional[str] = None


class GeminiEmbedContentRequest(BaseModel):
    content: GeminiContent
    taskType: Optional[str] = None
    title: Optional[str] = None
    outputDimensionality: Optional[int] = None


class GeminiBatchEmbedRequest(BaseModel):
    requests: list   # list of GeminiEmbedContentRequest-like dicts


GEMINI_TASK_MAP = {
    "RETRIEVAL_QUERY": "retrieval.query",
    "RETRIEVAL_DOCUMENT": "retrieval.passage",
    "SEMANTIC_SIMILARITY": "text-matching",
    "CLASSIFICATION": "classification",
    "CLUSTERING": "clustering",
    "QUESTION_ANSWERING": "retrieval.query",
    "FACT_VERIFICATION": "retrieval.query",
    "CODE_RETRIEVAL_QUERY": "retrieval.query",
}


def _parse_gemini_content(content_obj) -> list:
    """
    Parse a Gemini content object into ST-compatible inputs.
    Returns a list of parts (fused if multiple).
    """
    parts_raw = content_obj if isinstance(content_obj, list) else content_obj.get("parts", [])
    result = []
    for part in parts_raw:
        if isinstance(part, dict):
            if "inlineData" in part:
                result.extend(_parse_content_part(part))
            elif "text" in part:
                result.append(part["text"])
            else:
                result.extend(_parse_content_part(part))
        elif hasattr(part, "text") and part.text is not None:
            result.append(part.text)
        elif hasattr(part, "inlineData") and part.inlineData is not None:
            result.extend(_parse_content_part({"inlineData": part.inlineData}))
    return result


@app.post("/v1/models/{model_id}:embedContent", tags=["Google Gemini"])
async def embed_content_gemini(
    model_id: str,
    request: GeminiEmbedContentRequest,
):
    """
    Google Gemini-compatible single embedding endpoint.
    Supports text parts and inlineData (image/audio/video) parts.

    Compatible with:
    - genai.embed_content(model=..., content=..., task_type="RETRIEVAL_QUERY")
    """
    task = GEMINI_TASK_MAP.get(request.taskType or "", "retrieval")

    parts = _parse_gemini_content(request.content.dict())
    has_multimodal = any(not isinstance(p, str) for p in parts)

    if has_multimodal:
        _require_omni()
        item = parts if len(parts) > 1 else parts[0]
        embeddings, n_tokens, tok_per_s = _embed_mixed([item], task=task, dimensions=request.outputDimensionality)
    else:
        text = " ".join(str(p) for p in parts)
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
    Supports text and inlineData (multimodal) parts.

    Compatible with:
    - genai.batch_embed_contents(requests=[...])
    """
    all_items = []
    dims = None
    has_multimodal = False
    first_task_type = ""

    for req in request.requests:
        if not dims and req.get("outputDimensionality"):
            dims = req["outputDimensionality"]
        if not first_task_type:
            first_task_type = req.get("taskType", "")

        content = req.get("content", {})
        parts = _parse_gemini_content(content)

        if any(not isinstance(p, str) for p in parts):
            has_multimodal = True

        all_items.append(parts if len(parts) > 1 else (parts[0] if parts else ""))

    task = GEMINI_TASK_MAP.get(first_task_type, "retrieval")

    if has_multimodal:
        _require_omni()
        embeddings, n_tokens, tok_per_s = _embed_mixed(all_items, task=task, dimensions=dims)
    else:
        texts = [item if isinstance(item, str) else " ".join(str(p) for p in item) for item in all_items]
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
    documents: list
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
    with torch.inference_mode():
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
