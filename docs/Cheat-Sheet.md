One-page quick reference.

## Summary

> Deploy Jina AI embedding, reranker, and reader models on your own infrastructure with zero outbound network calls.

## The two phases

```
   PHASE 1                       PHASE 2
   bundle on connected           deploy on offline
   ───────────────────           ─────────────────

   bundle  ─►  .tar.gz  ─►  docker load  ─►  docker run  ─►  port 8080
   (network)   (USB / SCP)                  (offline)        5 schemas
```

## Deployment commands

```bash
# Phase 1 (connected machine):
./scripts/pull-prebuilt.sh jina-embeddings-v5-text-nano cpu
# or to build from source:
python jina-on-prem.py bundle --model MODEL --cpu-only --yes

# Phase 2 (offline machine):
docker load < MODEL.tar.gz
docker run -d -p 8080:8080 jina/MODEL:cpu
curl http://localhost:8080/health
```

## Headline models

| Model | Use case | Params | VRAM | Prebuilt |
|---|---|---|---|---|
| `jina-embeddings-v5-text-nano` | first demo, OpenAI replacement | 239M | ~2GB | yes |
| `jina-embeddings-v5-text-small` | production multilingual | 677M | ~3GB | yes |
| `jina-embeddings-v5-omni-nano` | multimodal RAG on a smaller GPU | 1.04B | ~5GB | yes |
| `jina-embeddings-v5-omni-small` | multimodal RAG | 1.74B | ~8GB | yes |
| `jina-reranker-v3.5` | top-K reranking, 131K context | 0.6B | ~3GB | yes |
| `jina-reranker-v3` | top-K reranking, lighter host RAM on `:cpu` | 597M | ~3GB | yes |
| `jina-clip-v2` | image + text search | 865M | ~4GB | yes |
| `jina-embeddings-v3` | in maintenance; new deployments should prefer `jina-embeddings-v5-text-small` ([Lifecycle](Product-And-Model-Lifecycle#current-lifecycle-status)) | 570M | ~3GB | yes |

Full table: [Model Catalog](Model-Catalog).

## Five API schemas (all simultaneous, same container)

| Schema | Endpoint | One-line example |
|---|---|---|
| Jina | `POST /v1/embeddings` | `{"input":["hi"],"task":"retrieval.query"}` |
| OpenAI | `POST /v1/embeddings` | `{"input":["hi"]}` |
| Cohere | `POST /v2/embed` | `{"texts":["hi"],"input_type":"search_query"}` |
| Gemini | `POST /v1/models/MODEL:embedContent` | `{"content":{"parts":[{"text":"hi"}]}}` |
| Voyage | `POST /v1/multimodalembeddings` | `{"inputs":[{"content":[{"type":"text","text":"hi"}]}]}` (multimodal models) |
| Reranker | `POST /v1/rerank`, `POST /v2/rerank` | `{"query":"...","documents":["...","..."],"top_n":2}` |
| Reader / VLM | `POST /v1/chat/completions` | `{"messages":[{"role":"user","content":"hi"}]}` |
| Health | `GET /health` | (returns model + device + schemas) |

## Tasks (v5)

`retrieval.query`, `retrieval.passage`, `text-matching` (default), `classification`, `clustering`. Pass via `task` field; any other value returns HTTP 400. Different task = different vector.

## Matryoshka truncation

`{"input":["hi"], "dimensions": 128}` → returns 128-dim vector. Any value up to the model's full output is accepted and truncated to; the Output column of the [Model Catalog](Model-Catalog) gives the smallest width each model was trained for.

## License

| License | Models | Commercial use |
|---|---|---|
| CC-BY-NC-4.0 | All v5, v3, reranker-v3 / -v3.5 / -m0 / -v2-base-multilingual, clip-v2, colbert-v2, code-embeddings, ReaderLM-v2, reader-lm, vlm | needs commercial license - contact [Elastic sales](https://www.elastic.co/contact) |
| Qwen Research License | v4 | needs commercial license - contact [Elastic sales](https://www.elastic.co/contact) |
| Apache-2.0 | v2 text embeddings (`jina-embeddings-v2-base-*`), v1 embeddings and rerankers, clip-v1, colbert-v1-en | free for any use |

## Hardware quick rules

- Low sustained QPS → CPU is fine; the cut-over is in [Sizing & Hardware -> Runtime: GPU or CPU?](Sizing-And-Hardware#runtime-gpu-or-cpu)
- Above that, or low-latency → L4 (24 GB VRAM, handles any single model)
- More than one L4 can serve → multiple replicas behind an LB, or A10G/A100; plan from [Throughput math](Sizing-And-Hardware#throughput-math)

## Pre-flight checklist

- [ ] Docker 24+ installed on target host
- [ ] (GPU) NVIDIA driver >= 525, NVIDIA Container Toolkit, `nvidia-smi` works
- [ ] 2-3x bundle size disk free
- [ ] Port 8080 (or chosen port) open
- [ ] Channel agreed for transferring the `.tar.gz`
- [ ] Decision: single host (POC), HA pair (prod), or k8s

## Key technical guarantees

1. Data and queries never leave the deployment host: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are baked into the image. The one outbound code path is the media URL fetch, reachable only by a multimodal request that carries an `http(s)` URL; see [Security & Hardening](Security-And-Hardening#the-one-outbound-path).
2. OpenAI-compatible: any client using `openai.OpenAI(base_url=...)` works without code changes.
3. The air-gap is verifiable: run on a host with no egress route and `docker logs` will be silent on outbound attempts. Source code on GitHub is Apache-2.0 and auditable.

## Common first-mile errors

| Error | Fix |
|---|---|
| `permission denied connecting to docker API` | `sudo usermod -aG docker $USER`, reconnect |
| `Error response: unauthorized` (on `docker pull`) | Not an auth problem - published images pull anonymously. No image exists at that name; check the Prebuilt column of the [Model Catalog](Model-Catalog) |
| `no matching manifest for linux/arm64/v8` | Apple Silicon: add `--platform linux/amd64` to `pull` and `run` |
| `bind: address already in use` | Map a different port: `-p 9090:8080` |
| Container exits with CUDA error | Driver too old: `nvidia-smi` should show >=525 |
| `OOM` on GPU | Check [Model Catalog](Model-Catalog) VRAM; pick smaller model or bigger GPU |

Full list: [Troubleshooting](Troubleshooting).

## Links to keep handy

- Wiki home: https://github.com/jina-ai/jina-on-prem/wiki
- Repo: https://github.com/jina-ai/jina-on-prem
- Issues: https://github.com/jina-ai/jina-on-prem/issues
- Commercial license: https://www.elastic.co/contact
