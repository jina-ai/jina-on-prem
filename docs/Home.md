# jina-airgap

Air-gapped deployment toolkit for Jina AI embedding, reranker, and reader models. Bundle a model into a self-contained Docker image on a connected machine, transfer the `.tar.gz`, run it offline on the air-gapped machine. No license server, no phone-home, no runtime network calls.

> **New here?** Start with the [Quick Start](Quick-Start) - first inference in under 5 minutes using a prebuilt image.

![overview](images/03-deploy.gif)

## Two phases

```mermaid
flowchart LR
    A[Bundle on networked machine] -->|USB / SCP / physical media| B[Deploy on air-gapped machine]
    B --> C[Multi-schema API ready]
    C --> D[Elasticsearch / your app]
```

1. **Bundle** (connected): `python jina-airgap.py bundle --model X` downloads weights and pinned deps, bakes them into a Docker image, exports as `.tar.gz`.
2. **Deploy** (offline): `docker load < X.tar.gz && docker run -p 8080:8080 jina/X` brings up a server that simultaneously speaks OpenAI, Cohere, Gemini, and Voyage AI API schemas.

If a prebuilt image already exists for your model, skip phase 1 entirely - see [Quick Start](Quick-Start).

## Wiki contents

- [Quick Start](Quick-Start) - 5-minute happy path using a prebuilt image
- [Bundling Guide](Bundling-Guide) - build your own bundle from a connected machine (GCP L4 walkthrough inside)
- [Model Catalog](Model-Catalog) - all 28 supported models with VRAM, context, output dims
- [API Reference](API-Reference) - the four API schemas, tasks, multimodal inputs
- [Troubleshooting](Troubleshooting) - common errors and the fixes that work

Contributing? See [CONTRIBUTING.md](https://github.com/jina-ai/jina-airgap/blob/main/CONTRIBUTING.md).

## Licensing

Most v5 / v4 / v3 models are **CC-BY-NC-4.0** - commercial use needs a license. Contact [Elastic sales](https://www.elastic.co/contact). v2 and v1 models are Apache-2.0 and free for commercial use. See the [Model Catalog](Model-Catalog) for per-model license.
