# NVIDIA NIM Deployment

Deploy Jina AI models via [NVIDIA NIM (Inference Microservices)](https://developer.nvidia.com/nim) — a containerized serving runtime with an OpenAI-compatible API, GPU kernel optimizations through vLLM, and CUDA graph capture.

This is an alternative to the standard `jina-airgap` bundle/deploy workflow. It trades the fully-offline bundle-everything approach for a connected pull from NGC (NVIDIA GPU Cloud) at startup, with model weights cached locally for subsequent runs.

---

## When to use NIM vs the standard airgap bundle

| | Standard airgap bundle | NVIDIA NIM |
|---|---|---|
| **Network at deploy time** | Not required | NGC auth required on first pull |
| **GPU required** | No (CPU builds available) | Yes (NVIDIA GPU + CUDA) |
| **API compatibility** | OpenAI + Cohere + Voyage + Gemini | OpenAI only |
| **Serving engine** | sentence-transformers (FastAPI) | vLLM (optimized) |
| **Model weights delivery** | Baked into Docker image | Downloaded to cache volume |
| **CUDA graph optimization** | No | Yes |
| **Best for** | True air-gap, CPU-only, custom API schemas | GPU inference, OpenAI-compatible tooling, NVIDIA ecosystem |

---

## Prerequisites

- NVIDIA GPU with CUDA 12.0+ (tested on L40S 46GB)
- Docker with NVIDIA Container Toolkit (`nvidia-container-runtime`)
- NVIDIA NGC API key — get one free at [ngc.nvidia.com](https://ngc.nvidia.com/setup)
- ~5–10 GB disk per model for the weight cache

```bash
# Verify GPU is visible to Docker
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

---

## Quickstart

### 1. Log in to NGC

```bash
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

### 2. Run a model

```bash
docker run -d --name jina-embed \
  --gpus all --shm-size=16GB \
  -p 8000:8000 \
  -v $HOME/nim_cache:/opt/nim/.cache \
  -e NGC_API_KEY=$NGC_API_KEY \
  nvcr.io/nim/nvidia/model-free-nim:2.0.6 \
  hf://jinaai/jina-embeddings-v3 --trust-remote-code
```

### 3. Check health and call the API

```bash
# Wait for ready (typically 30–90 seconds for weight download + CUDA graph capture)
until curl -sf http://localhost:8000/v1/health/ready; do sleep 5; done

# Embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "ga-model-free-nim", "input": "hello world"}'

# Reranking (jina-reranker-v3 only — use /rerank not /v1/rerank)
curl -X POST http://localhost:8002/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ga-model-free-nim",
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a type of artificial intelligence.",
      "The weather today is sunny."
    ]
  }'

# Text generation (ReaderLM-v2)
curl -X POST http://localhost:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ga-model-free-nim",
    "messages": [{"role": "user", "content": "Convert to markdown: <h1>Hello</h1>"}],
    "max_tokens": 500
  }'
```

---

## Model compatibility

Tested with `nvcr.io/nim/nvidia/model-free-nim:2.0.6` (vLLM 0.21.0) on an NVIDIA L40S (46 GB):

| Model | HF repo | Type | NIM status | trust-remote-code | API endpoint |
|---|---|---|---|---|---|
| jina-embeddings-v3 | `jinaai/jina-embeddings-v3` | Embedding | ✅ Works | Required | `/v1/embeddings` |
| jina-embeddings-v5-text-small | `jinaai/jina-embeddings-v5-text-small` | Embedding | ✅ Works | Required | `/v1/embeddings` |
| jina-reranker-v3 | `jinaai/jina-reranker-v3` | Reranker | ✅ Works | Required | `/rerank` |
| ReaderLM-v2 | `jinaai/ReaderLM-v2` | Small LM | ✅ Works | Required | `/v1/chat/completions` |
| jina-embeddings-v5-omni-small | `jinaai/jina-embeddings-v5-omni-small` | Multimodal | ❌ Blocked | — | — |
| jina-embeddings-v5-omni-nano | `jinaai/jina-embeddings-v5-omni-nano` | Multimodal | ❌ Blocked | — | — |

**Why `--trust-remote-code` is always required**: all Jina models use `jinaai/xlm-roberta-flash-implementation` as a custom module, loaded even when the primary architecture (e.g. `Qwen2ForCausalLM`) is natively registered in vLLM.

**Why the omni models are blocked**: `JinaEmbeddingsV5OmniModel` is not registered in vLLM 0.21.0's model registry. NIM's profile selector raises `ValueError: Unable to detect a supported backend for the model` before loading. This will be resolved in a future NIM release when the architecture is upstreamed to vLLM.

### Reranker API note

NIM exposes the reranker at `/rerank` (not `/v1/rerank`). The `/v1/rerank` path returns 404. Use:

```bash
# Correct
POST http://localhost:<port>/rerank

# Wrong (404)
POST http://localhost:<port>/v1/rerank
```

---

## Running multiple models

Use port-per-model mapping. Each container needs its own GPU memory slice — on a 46 GB L40S, embedding and reranker models (~3 GB each) can run concurrently:

```bash
# Embeddings on 8000
docker run -d --name jina-embed \
  --gpus all --shm-size=16GB -p 8000:8000 \
  -v $HOME/nim_cache:/opt/nim/.cache \
  -e NGC_API_KEY=$NGC_API_KEY \
  nvcr.io/nim/nvidia/model-free-nim:2.0.6 \
  hf://jinaai/jina-embeddings-v3 --trust-remote-code

# Reranker on 8001
docker run -d --name jina-reranker \
  --gpus all --shm-size=16GB -p 8001:8000 \
  -v $HOME/nim_cache:/opt/nim/.cache \
  -e NGC_API_KEY=$NGC_API_KEY \
  nvcr.io/nim/nvidia/model-free-nim:2.0.6 \
  hf://jinaai/jina-reranker-v3 --trust-remote-code

# ReaderLM on 8002
docker run -d --name jina-reader \
  --gpus all --shm-size=16GB -p 8002:8000 \
  -v $HOME/nim_cache:/opt/nim/.cache \
  -e NGC_API_KEY=$NGC_API_KEY \
  nvcr.io/nim/nvidia/model-free-nim:2.0.6 \
  hf://jinaai/ReaderLM-v2 --trust-remote-code
```

---

## Key flags reference

| Flag / env var | Required | Where to set | Purpose |
|---|---|---|---|
| `NGC_API_KEY` | Yes | `-e NGC_API_KEY=...` | Authentication with NGC to pull the NIM runtime and download model metadata |
| `--trust-remote-code` | Yes (all Jina models) | Positional CLI arg after the image name | Allows vLLM to load `jinaai/xlm-roberta-flash-implementation` custom code |
| `hf://jinaai/<model>` | Yes | Positional CLI arg (first) | Tells `model-free-nim` which HuggingFace model to serve |
| `--gpus all` | Yes | Docker run flag | Passes GPU(s) into the container |
| `--shm-size=16GB` | Recommended | Docker run flag | Shared memory for vLLM worker communication |
| `-v $HOME/nim_cache:/opt/nim/.cache` | Recommended | Volume mount | Caches downloaded weights across container restarts |
| `NIM_MODEL_PATH` | Alternative | `-e NIM_MODEL_PATH=hf://...` | Alternative to positional `hf://` arg — sets model path via env var |

> **Note**: `NIM_FORCE_TRUST_REMOTE_CODE=1` works for `llm-nim` but is **not recognized** by `model-free-nim`. You must pass `--trust-remote-code` as a positional CLI argument after the image name.

---

## Startup time and caching

On first launch, `model-free-nim` downloads model weights from HuggingFace (~2–4 GB per model) into `/opt/nim/.cache`. Mounting a host directory at that path persists the cache:

```bash
-v $HOME/nim_cache:/opt/nim/.cache
```

Subsequent restarts skip the download. Typical startup after first cache:
- Embedding models: ~30–60 seconds (CUDA graph capture)
- ReaderLM-v2: ~60–90 seconds

---

## Fully air-gapped NIM deployment

If your target environment has **no internet access at all** (no NGC, no HuggingFace), you need to pre-download both the NIM container and model weights on a connected machine and transfer them in.

### Phase 1 — online prep (connected machine)

```bash
# 1. Pull and export the NIM container image
docker pull nvcr.io/nim/nvidia/model-free-nim:2.0.6
docker save nvcr.io/nim/nvidia/model-free-nim:2.0.6 | gzip > model-free-nim-2.0.6.tar.gz

# 2. Pre-download model weights into a cache directory
mkdir -p $HOME/nim_cache
docker run --rm --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v $HOME/nim_cache:/opt/nim/.cache \
  nvcr.io/nim/nvidia/model-free-nim:2.0.6 \
  hf://jinaai/jina-embeddings-v3 --trust-remote-code &
# Wait for the model to finish loading, then stop the container
# The weights are now in $HOME/nim_cache

# 3. Archive and transfer to the air-gapped machine
tar -czf nim_cache.tar.gz -C $HOME nim_cache
# Transfer model-free-nim-2.0.6.tar.gz and nim_cache.tar.gz via USB/SCP/physical media
```

### Phase 2 — offline deploy (air-gapped machine)

```bash
# Load the pre-transferred assets
docker load < model-free-nim-2.0.6.tar.gz
tar -xzf nim_cache.tar.gz -C $HOME

# Run without NGC_API_KEY — all weights are local
docker run -d --name jina-embed \
  --gpus all --shm-size=16GB \
  -p 8000:8000 \
  -v $HOME/nim_cache:/opt/nim/.cache \
  -e NIM_MODEL_PATH=hf://jinaai/jina-embeddings-v3 \
  nvcr.io/nim/nvidia/model-free-nim:2.0.6 \
  hf://jinaai/jina-embeddings-v3 --trust-remote-code
```

> **Note**: Do not set `NGC_API_KEY` or `HF_TOKEN` in the air-gapped phase — the container will try to reach out to verify them and fail. Omitting them forces fully local loading from the cache volume.

For detailed guidance on NIM air-gap workflows (profile selection, model stores, `download-to-cache` subcommand), see the official NVIDIA documentation:
**https://docs.nvidia.com/nim/large-language-models/latest/deployment/air-gap-deployment.html**

---

## Comparing NIM to the standard bundle workflow

The standard `jina-airgap bundle` bakes weights into a Docker image, making the deploy phase truly offline with no runtime steps required. NIM requires a pre-download step but provides:

- Better GPU throughput via vLLM's CUDA graph optimization
- Easy model switching without rebuilding images
- OpenAI-compatible API out of the box

For true offline deploys with CPU support and multiple API schema compatibility (Cohere, Voyage, Gemini), use the standard bundle workflow. For GPU-optimized inference in NVIDIA-equipped environments, NIM provides better throughput.
