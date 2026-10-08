A deployed jina-on-prem server exposes five API schemas on the same port. Pick whichever your client already speaks - they all hit the same model.

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

| Schema | Endpoint | Drop-in for |
|---|---|---|
| Jina | `POST /v1/embeddings`, `POST /v1/rerank` | the hosted Jina API, byte-compatible |
| OpenAI | `POST /v1/embeddings` | OpenAI SDK, Elasticsearch inference, LlamaIndex, LangChain |
| Voyage AI | `POST /v1/embeddings` (with `input_type` / `output_dimension`), `POST /v1/multimodalembeddings` | Voyage SDK |
| Cohere | `POST /v2/embed`, `POST /v2/rerank` | Cohere SDK |
| Google Gemini | `POST /v1/models/{model}:embedContent`, `:batchEmbedContents` | Google AI SDK |
| Utility | `GET /health` | health checks |

The Cohere schema is served on its v2 paths, which is what a current Cohere SDK calls.

The reader and VLM models serve `POST /v1/chat/completions` instead of an embedding endpoint - they generate text. Each model's `api_endpoint` field in the catalog says which endpoint it answers on, and so does `GET /health`.

## Health

```bash
curl http://localhost:8080/health
```
```json
{"status":"ok","model":"jina-embeddings-v5-text-nano","category":"text","device":"cpu","ready":true,"multimodal":false,"endpoint":"/v1/embeddings","schemas":["jina","openai","voyage","gemini","cohere"],"license":{"mode":"warn","valid":false,"reason":"no_license","fail_open":true}}
```

## OpenAI / Voyage

```bash
curl -s http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["Hello world"]}'
```

Optional fields:

| Field | Schema | Notes |
|---|---|---|
| `model` | OpenAI / Voyage | accepted; echoed in response. The actual model is fixed by the container |
| `task` | (extension) | `retrieval.query`, `retrieval.passage`, `text-matching`, `classification`, `clustering` for v5; `retrieval.query`, `retrieval.passage`, `text-matching`, `classification`, `separation` for v3. A value outside the model's task list returns HTTP 400 naming the accepted values |
| `input_type` | Voyage | `query` / `document` - mapped to `task` automatically |
| `dimensions` | OpenAI | matryoshka truncation - any value up to the model's full output is accepted; the smallest trained width is per model, see the [Model Catalog](Model-Catalog) |
| `output_dimension` | Voyage | alias for `dimensions` |

Python:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
resp = client.embeddings.create(model="jina-embeddings-v5-text-nano", input=["Hello world"])
```

Matryoshka:
```bash
curl -s http://localhost:8080/v1/embeddings \
  -d '{"input":["Hello"], "dimensions": 128}' \
  -H 'Content-Type: application/json' \
  | jq '.data[0].embedding | length'
# 128
```

## Cohere

```bash
curl -s http://localhost:8080/v2/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["Hello world"], "input_type": "search_query"}'
```

Response shape (matches Cohere's `/v2/embed`):
```json
{
  "id": "...",
  "texts": ["Hello world"],
  "embeddings": {"float": [[ ... 768 floats ... ]]},
  "images": [],
  "meta": {...}
}
```

`input_type` -> `task` mapping:
- `search_query`, `query` -> retrieval (query side)
- `search_document`, `document` -> retrieval (passage side)
- `classification` -> classification
- `clustering` -> clustering (`separation` on v3)
- `image` -> no task; it goes with `images`, whose modality already says what the input is

## Gemini

```bash
# single
curl -s "http://localhost:8080/v1/models/jina-embeddings-v5-text-nano:embedContent" \
  -H 'Content-Type: application/json' \
  -d '{"content": {"parts": [{"text": "Hello"}]}, "taskType": "RETRIEVAL_QUERY"}'

# batch
curl -s "http://localhost:8080/v1/models/jina-embeddings-v5-text-nano:batchEmbedContents" \
  -H 'Content-Type: application/json' \
  -d '{"requests": [
    {"content": {"parts": [{"text": "first"}]}},
    {"content": {"parts": [{"text": "second"}]}}
  ]}'
```

Response uses Gemini's shape: `{"embedding": {"values": [...]}}` for single, `{"embeddings": [...]}` for batch.

`taskType` -> `task` mapping, where a value lists several tasks the first one the loaded model publishes wins:
- `TASK_TYPE_UNSPECIFIED` -> no task; the model's default
- `RETRIEVAL_QUERY` -> retrieval (query side)
- `RETRIEVAL_DOCUMENT` -> retrieval (passage side)
- `SEMANTIC_SIMILARITY` -> text-matching
- `CLASSIFICATION` -> classification
- `CLUSTERING` -> clustering, or separation
- `QUESTION_ANSWERING` -> qa, or retrieval (query side)
- `FACT_VERIFICATION` -> retrieval (query side)
- `CODE_RETRIEVAL_QUERY` -> nl2code, code, or retrieval (query side)

## Reranker

```bash
curl -s http://localhost:8080/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "best programming language for AI",
    "documents": [
      "Python is the most popular language for ML",
      "Bananas are yellow",
      "PyTorch and TensorFlow use Python"
    ],
    "top_n": 2
  }'
```

Response (Jina schema):
```json
{
  "model": "jina-reranker-v3.5",
  "object": "list",
  "usage": {"total_tokens": 42},
  "results": [
    {"index": 0, "relevance_score": 0.21, "document": {"text": "Python is the most popular..."}},
    {"index": 2, "relevance_score": -0.003, "document": {"text": "PyTorch and TensorFlow..."}}
  ]
}
```

Timing is reported in the `X-Jina-Elapsed-Ms` and `X-Jina-Tok-Per-S` response headers, not in the body. The Cohere response shape is served at `/v2/rerank`.

> Every container mounts all five schemas, but only the ones its model can serve return a result: a reranker answers `/v1/rerank` and `/v2/rerank`, and returns HTTP 400 on `/v1/embeddings` with a message naming the endpoint to use. Run one container per role.

## Multimodal inputs (omni / clip / v4 / vlm)

Multimodal models accept images, audio, and video alongside text. Media can be sent as inline base64, as a `data:` URL, or as an `http(s)` URL that the server fetches under a size cap, a timeout and a redirect limit, the constants at the top of `server/media.py`. In an air-gapped network a URL fetch fails, so send media inline there. Max 10 MB per input.

```bash
# Image-only embedding
curl -s http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": [{"type": "image_base64",
                   "image_base64": {"base64": "<B64>", "mime_type": "image/png"}}]}'

# Fused text + image -> single embedding
curl -s http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": [{"content": [
          {"type": "text", "text": "A red square"},
          {"type": "image", "format": "base64", "value": "<B64>"}
       ]}]}'
```

Text-only models return HTTP 400 with a helpful list of multimodal-capable model IDs if you send multimodal input.

## Embedding tasks

`task` is the high-leverage field. Embeddings optimized for retrieval differ from those for classification or clustering.

| Model family | Supported tasks |
|---|---|
| v5-text / v5-omni | `retrieval.query`, `retrieval.passage`, `text-matching` (default), `classification`, `clustering` |
| v4 | `retrieval.query`, `retrieval.passage`, `text-matching` (default), `code.query`, `code.passage` |
| code-embeddings | `nl2code.query` (default), `nl2code.passage`, and `.query` / `.passage` forms of `qa`, `code2code`, `code2nl`, `code2completion` |
| v3 | `retrieval.query`, `retrieval.passage`, `text-matching`, `classification`, `separation` |
| v2 / v1 | no task field; passed values are ignored |

The same model called with `retrieval.query` vs `retrieval.passage` (v3) returns **different** vectors - this is the asymmetric retrieval pattern.

## Elasticsearch integration

Register embeddings as `service: openai` pointed at `/v1/embeddings`, and rerankers as `service: cohere` pointed at `/v1/rerank` or `/v2/rerank`. Newer Elasticsearch versions, 9.5.3 among them, always call `/v2/rerank`, so a proxy in front has to pass it through. [Elasticsearch Integration](Elasticsearch-Integration) has the full walkthrough: endpoint registration, `semantic_text`, reranking, and running more than one replica behind a single URL.

## Auto-generated docs

Run the server and hit `GET /docs` for the FastAPI-generated Swagger UI - it's always in sync with the deployed code.

## Source of truth

Full request/response shapes: the adapters in [`server/adapters/`](https://github.com/jina-ai/jina-on-prem/tree/main/server/adapters), one module per schema.
