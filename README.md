# jina-airgapped

Air-gapped deployment toolkit for Jina AI models. Ship embedding, reranker, and reader models to fully disconnected environments.

## Why

- Customers in regulated/air-gapped environments (gov, finance, healthcare)
- No NVIDIA NIM ($4,500/GPU/yr overkill for embedding models)
- All Jina models fit on a single L4 GPU (~$0.80/hr)
- OpenAI-compatible API - drop-in for Elasticsearch inference service

## Quick Start

### On a connected machine

```bash
# List available models
python jina-airgapped.py list

# Interactive wizard: select model, build Docker image, save to .tar.gz
python jina-airgapped.py pack

# Or specify directly
python jina-airgapped.py pack --model jina-embeddings-v3 --output jina-emb-v3.tar.gz

# For gated models (jina-embeddings-v4 uses Qwen license)
python jina-airgapped.py pack --model jina-embeddings-v4 --hf-token hf_xxx

# CPU-only (no GPU at runtime)
python jina-airgapped.py pack --model jina-embeddings-v3 --cpu-only
```

### On the air-gapped machine

```bash
# Transfer the .tar.gz file, then:
docker load < jina-emb-v3.tar.gz
docker run --gpus all -p 8080:8080 jina/jina-embeddings-v3:gpu

# CPU only
docker run -p 8080:8080 jina/jina-embeddings-v3:cpu

# Or use the helper
python jina-airgapped.py load --image jina-emb-v3.tar.gz --gpu
```

### Test it

```bash
# Health check
curl http://localhost:8080/health

# Embedding
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello world", "Jina AI"], "model": "jina-embeddings-v3"}'

# Rerank
curl -X POST http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "documents": ["AI is cool", "Python rocks", "ML is ML"]}'

# Reader (HTML to Markdown)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "<html><body><h1>Hi</h1></body></html>"}]}'
```

## Available Models

| Model | Type | Params | VRAM | Context |
|-------|------|--------|------|---------|
| jina-embeddings-v3 | text embedding | 570M | ~3GB | 8K |
| jina-embeddings-v4 | multimodal embedding | 3.8B | ~8GB | 32K |
| jina-embeddings-v5-text-nano | text embedding | 239M | ~2GB | 8K |
| jina-embeddings-v5-text-small | text embedding | 677M | ~3GB | 32K |
| jina-embeddings-v5-omni-nano | multimodal embedding | 1.04B | ~5GB | 8K |
| jina-embeddings-v5-omni-small | multimodal embedding | 1.74B | ~8GB | 32K |
| jina-clip-v2 | multimodal embedding | 865M | ~4GB | 8K |
| jina-reranker-v3 | reranker | 597M | ~3GB | 131K |
| ReaderLM-v2 | reader/LLM | 1.54B | ~4GB | 512K |

All models tested on single L4 GPU (24GB VRAM). Zero phone-home, no license server.

## Elasticsearch Integration

The API is OpenAI-compatible. Use the `openai` inference service type in Elasticsearch:

```json
PUT _inference/text_embedding/jina-local
{
  "service": "openai",
  "service_settings": {
    "url": "http://your-host:8080/v1/embeddings",
    "model_id": "jina-embeddings-v3",
    "api_key": "not-needed"
  }
}
```

## Serve Without Docker

If the model dependencies are already installed:

```bash
python jina-airgapped.py serve --model jinaai/jina-embeddings-v3 --type embedding --port 8080

# From local path
python jina-airgapped.py serve --local-path /data/models/jina-emb-v3 --type embedding
```

## Design

- **Zero deps for TUI**: `jina-airgapped.py` uses Python stdlib only for the UI
- **Model baked in**: `HF_HUB_OFFLINE=1` enforced at runtime, no downloads
- **Multi-stage build**: small runtime image, weights in layer
- **GPU auto-detect**: container falls back to CPU if no GPU
- **OpenAI API**: drop-in for any client expecting OpenAI format
- **Matryoshka support**: pass `dimensions` to truncate embeddings

## Repo Structure

```
jina-airgapped/
├── README.md
├── jina-airgapped.py     # Main TUI tool (zero external deps for UI)
├── models/
│   └── catalog.json      # Model registry
├── docker/
│   ├── embeddings/Dockerfile
│   ├── reranker/Dockerfile
│   └── reader-lm/Dockerfile
├── server/
│   ├── app.py            # FastAPI inference server
│   └── requirements.txt
└── tests/
    └── test_e2e.py       # E2E tests
```
