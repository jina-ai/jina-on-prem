# Architecture

How the pieces fit together. Useful for SAs answering "what is this thing made of?" and for engineers debugging.

## The two-phase model

```mermaid
flowchart TB
    classDef bundle fill:#fff3d6,stroke:#c08800
    classDef deploy fill:#d9f5e0,stroke:#1f8f3a
    classDef hop fill:#e8f0ff,stroke:#3b6ad6

    subgraph P1["Phase 1: BUNDLE (connected machine)"]
        direction TB
        CLI[python jina-airgap.py bundle] --> CAT[Read models/catalog.json
get pinned deps + hf_repo]
        CAT --> BLD[DOCKER_BUILDKIT=1 docker build
Dockerfile.cpu or Dockerfile.gpu]
        BLD --> DL[Stage 1: download_model.py
HF Hub download
patch model code
delete model repo requirements.txt]
        DL --> RT[Stage 2: install pinned deps
copy weights from stage 1
add server/app.py
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1]
        RT --> SAVE[docker save MODEL | gzip > MODEL.tar.gz]
    end
    class CLI,CAT,BLD,DL,RT,SAVE bundle

    SAVE --> HOP[Transfer .tar.gz
USB / SCP / approved channel]
    class HOP hop

    subgraph P2["Phase 2: DEPLOY (offline machine)"]
        direction TB
        LD[docker load < MODEL.tar.gz] --> RUN[docker run -d -p 8080:8080 jina/MODEL]
        RUN --> SRV[FastAPI on 8080
multi-schema API]
        SRV --> APP[Customer app
ES / LlamaIndex / curl]
    end
    class LD,RUN,SRV,APP deploy
```

## What's in the Docker image

```mermaid
flowchart TB
    subgraph Image["jina/MODEL:cpu or :gpu"]
        OS[Base OS layer
python:3.11-slim CPU
or pytorch/pytorch:2.5.1-cuda12.1 GPU]
        DEPS[Pinned Python deps
from models/catalog.json deps]
        WEIGHTS[Model weights and tokenizer
~500MB to ~10GB]
        CODE[Custom model code
patched for offline use]
        SERVER[server/app.py + requirements.txt
FastAPI multi-schema API]
        ENV[Env: HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1]
    end

    Image -->|docker run| Container[Running container
HTTP server on 8080]
```

Critical detail: **the model's own `requirements.txt`** (shipped in some HF repos) is deleted at build time by `docker/download_model.py`. This prevents the model from auto-upgrading transformers to a newer version at runtime, which would break it. The pinned deps win.

## The multi-schema server

```mermaid
flowchart LR
    R1[OpenAI client] -->|POST /v1/embeddings| SRV
    R2[Voyage client] -->|POST /v1/embeddings input_type=...| SRV
    R3[Cohere client] -->|POST /v1/embed| SRV
    R4[Gemini client] -->|POST /v1/models/X:embedContent| SRV
    R5[Reranker client] -->|POST /v1/rerank| SRV
    R6[ES inference] -->|service openai or cohere| SRV

    SRV[FastAPI handlers
server/app.py] --> ROUTE{schema-specific
adapters}
    ROUTE --> ENC[Model encode
SentenceTransformer / CrossEncoder]
    ENC --> SHAPE[Shape response
per schema]
    SHAPE --> R1
    SHAPE --> R2
    SHAPE --> R3
    SHAPE --> R4
    SHAPE --> R5
    SHAPE --> R6
```

One container, four protocols. The shape adapter is the only schema-aware code; the encode call is shared.

## How weights stay private

```mermaid
sequenceDiagram
    participant Build as Build (connected)
    participant HF as HuggingFace Hub
    participant Image as Docker image
    participant Deploy as Deploy (offline)
    participant App as Customer app

    Build->>HF: download weights, tokenizer, code
    Build->>Build: patch code for offline
    Build->>Build: delete model's requirements.txt
    Build->>Image: bake weights + deps + env
    Note over Image: HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

    Image->>Deploy: transfer .tar.gz (one-time)
    Deploy->>Deploy: docker load + docker run
    App->>Deploy: POST /v1/embeddings
    Deploy->>Deploy: encode locally
    Deploy->>App: vectors
    Note over Deploy,App: zero outbound calls
ever
```

At runtime, any code path that would try to download a missing file fails immediately because of the env vars. The image is self-sufficient.

## CLI structure

```
jina-airgap.py (single file, stdlib only)
- cmd_list      - browse models/catalog.json
- cmd_bundle    - read catalog, write deps file, run docker build, save tar.gz
- cmd_deploy    - docker load + docker run + health check
- cmd_serve     - run server/app.py directly (skip Docker, if deps installed)
```

No third-party imports in the CLI. The model deps install inside the Docker image only.

## File layout

```
jina-airgap/
- jina-airgap.py             CLI (stdlib only)
- models/
  - catalog.json             28 models with pinned deps, vram, modality
- docker/
  - Dockerfile.cpu           two-stage: downloader + runtime
  - Dockerfile.gpu           two-stage with CUDA + cudnn
  - download_model.py        runs in stage 1, downloads + patches
- server/
  - app.py                   FastAPI with 4 schemas + /v1/rerank
  - requirements.txt         server framework deps
- scripts/
  - bootstrap-gcp.sh         provision a builder
  - pull-prebuilt.sh         pull GHCR image and tar.gz it
  - gen_catalog_md.py        regenerate the Model Catalog wiki page
  - sync-wiki.sh             push docs/ to the wiki
  - benchmark.py             throughput micro-benchmark
- docs/                      mirror of this wiki, edited via PRs
- tests/test_e2e.py          E2E API tests
- test_airgap.sh             quick smoke test for a built image
```

## What's intentionally NOT in scope

- **LLM chat / generation** (with a few exceptions like ReaderLM and jina-vlm). Use Ollama, vLLM, or LocalAI for chat models.
- **Distributed inference** across multiple GPUs. Each container is a single GPU. For more throughput, run more containers.
- **Fine-tuning**. Bundles serve as-is.
- **Streaming responses**. Embeddings and reranking are batch operations.
- **Auth / RBAC** at the API layer. Add this at your load balancer / API gateway.

## Next

- [API Reference](API-Reference) - exact request/response shapes per schema
- [Sizing & Hardware](Sizing-And-Hardware) - capacity planning
- [Troubleshooting](Troubleshooting) - debugging the pieces above
