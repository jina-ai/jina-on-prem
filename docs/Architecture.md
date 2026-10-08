How the pieces fit together, for engineers integrating or debugging jina-on-prem.

![mascot](images/pixel-mascot-shield.png)

## The two-phase model

```
  ┌─────────────────────────────────────────────────────────────────┐
  │ PHASE 1: BUNDLE (connected machine)                             │
  │                                                                 │
  │  python jina-on-prem.py bundle                                   │
  │         │                                                       │
  │         ▼                                                       │
  │  read models/catalog.json  ─►  pinned deps + hf_repo            │
  │         │                                                       │
  │         ▼                                                       │
  │  DOCKER_BUILDKIT=1 docker build                                 │
  │  │   stage 1 (builder):    install pinned deps into /opt/venv   │
  │  │   stage 2 (downloader): on the builder, download weights     │
  │  │                          from HF Hub, patch model code,      │
  │  │                          delete model repo requirements.txt  │
  │  │   stage 3 (runtime):    COPY only, no RUN, distroless        │
  │  │                          on the official base: venv from 1,  │
  │  │                          weights from 2, server/,            │
  │  │                          HF_HUB_OFFLINE=1                    │
  │  ▼                                                              │
  │  docker save | gzip > MODEL.tar.gz                              │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼  USB / SCP / approved channel
                                 │
  ┌──────────────────────────────┴──────────────────────────────────┐
  │ PHASE 2: DEPLOY (offline machine)                               │
  │                                                                 │
  │  docker load < MODEL.tar.gz                                     │
  │         │                                                       │
  │         ▼                                                       │
  │  docker run -d -p 8080:8080 jina/MODEL                          │
  │         │                                                       │
  │         ▼                                                       │
  │  FastAPI on :8080  ─►  your app (ES / LlamaIndex / curl)        │
  │  multi-schema API                                               │
  └─────────────────────────────────────────────────────────────────┘
```

## What's in the Docker image

```
  jina/MODEL:cpu  (or :gpu)
  ┌─────────────────────────────────────────────────────────┐
  │  ENV         HF_HUB_OFFLINE=1   TRANSFORMERS_OFFLINE=1  │
  ├─────────────────────────────────────────────────────────┤
  │  SERVER      server/: app.py, adapters/, families/      │
  │              FastAPI multi-schema API                   │
  ├─────────────────────────────────────────────────────────┤
  │  CODE        custom model code, patched for offline     │
  ├─────────────────────────────────────────────────────────┤
  │  WEIGHTS     model weights + tokenizer   (500 MB - 10 GB)│
  ├─────────────────────────────────────────────────────────┤
  │  DEPS        pinned Python deps from catalog.json       │
  ├─────────────────────────────────────────────────────────┤
  │  BASE        Wolfi, distroless in prebuilt images       │
  │              + CUDA 12.6 from PyTorch wheels (GPU)      │
  └─────────────────────────────────────────────────────────┘
              │
              │  docker run
              ▼
  HTTP server on :8080
```

Critical detail: **the model's own `requirements.txt`** (shipped in some HF repos) is deleted at build time by `docker/download_model.py`. This prevents the model from auto-upgrading transformers to a newer version at runtime, which would break it. The pinned deps win.

## The multi-schema server

```
   client                        endpoint
   ────────────────────────────────────────────────────────────────
   Jina API client        ──►  POST /v1/embeddings           ┐
   OpenAI SDK             ──►  POST /v1/embeddings           │
   Voyage SDK             ──►  POST /v1/multimodalembeddings │
   Cohere SDK             ──►  POST /v2/embed                ├──►  FastAPI server
   Google AI SDK          ──►  POST /v1/models/X:embedContent│         │
   reranker client        ──►  POST /v1/rerank, /v2/rerank   │         ▼
   reader / VLM client    ──►  POST /v1/chat/completions     │   model.encode()
   ES inference           ──►  service: openai | cohere      ┘
                                                                       │
   GET /health  (status, schemas, multimodal flag)                     ▼
                                                              schema-specific
                                                              response shaper
                                                                       │
                                                                       ▼
                                                                   reply
```

One container, five protocols. Every schema's router is mounted in every container; which endpoints return a result depends on the model it loaded. The shape adapter is the only schema-aware code; the encode call is shared.

## How weights stay private

```mermaid
sequenceDiagram
    participant Build as Build (connected)
    participant HF as Hugging Face Hub
    participant Image as Docker image
    participant Deploy as Deploy (offline)
    participant App as Your app

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
    Note over Deploy,App: no outbound calls, except a media URL carried by a request
ever
```

At runtime, any code path that would try to download a missing file fails immediately because of the env vars. The image is self-sufficient.

## CLI structure

```
jina-on-prem.py (single file, stdlib only)
- list      - browse the model catalog
- bundle    - build an image and save it as a transferable tar.gz
- deploy    - load the tar.gz, run it, print the health check command
- serve     - run the server directly, without Docker
- keygen    - mint an offline license key
```

No third-party imports in the CLI. The model deps install inside the Docker image only.

## File layout

```
jina-on-prem/
- jina-on-prem.py             CLI (stdlib only)
- models/
  - catalog.json             every model, with pinned deps, vram, modality
- docker/
  - Dockerfile.cpu           three stages: builder, downloader, COPY-only runtime
  - Dockerfile.gpu           the same, with CUDA from the PyTorch cu126 wheels
  - download_model.py        runs in the downloader stage, downloads + patches
  - verify_staging.py        build gate: trust_remote_code files already staged
- server/
  - app.py                   FastAPI app: mounts the adapters, license gate, /health
  - adapters/                one module per wire contract: Jina, OpenAI, Voyage, Cohere, Gemini
  - families/                one class per model runtime: load, task mapping, inference
  - engine.py                the loaded model's lifecycle and the inference verbs
  - batching.py              cross-request batching for the embedding path
  - tasks.py                 task vocabularies, ours and the other vendors'
  - license.py               offline license key check
  - telemetry.py             /health throughput block and X-Jina-* headers
  - (other modules)          config, catalog lookup, media decoding, serialization, errors
  - requirements.txt         deps for `serve` mode; images pin their own in the Dockerfiles
- k8s/
  - jina-on-prem.yaml        Kubernetes manifest: Deployment, Service, HPA, optional Ingress
- examples/
  - python_client.py         minimal client for the common request patterns
- scripts/
  - bootstrap-gcp.sh         provision a builder
  - pull-prebuilt.sh         pull GHCR image and tar.gz it
  - benchmark.py             throughput micro-benchmark
- docker-compose.yml         one model
- docker-compose.multi.yml   embedding + reranker, two containers
- docker-compose.replicas.yml  replicas of one model behind nginx, one port
- nginx-replicas.conf        nginx config for the replicas file
- docs/                      the source of these wiki pages
- verify-offline.sh          prove an image serves with no network at all
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
