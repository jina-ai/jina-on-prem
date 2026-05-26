# API Reference

A deployed jina-airgap server exposes four API schemas on the same port simultaneously. Pick whichever your client already speaks - they all hit the same underlying model.

| Schema | Endpoint | Drop-in for |
|---|---|---|
| OpenAI | `POST /v1/embeddings` | OpenAI SDK, Elasticsearch inference, LlamaIndex, LangChain |
| Cohere | `POST /v1/embed` | Cohere SDK, anything that speaks Cohere |
| Google Gemini | `POST /v1/models/{model}:embedContent` | Google AI SDK |
| Voyage AI multimodal | `POST /v1/multimodalembeddings` | Voyage SDK |
| (utility) | `GET /health` | health checks |
| (utility) | `POST /v1/rerank` | reranker models |

![api-schemas](images/04-schemas.gif)

## Health

```bash
curl http://localhost:8080/health
```
```json
{"status": "ok", "model": "jina-embeddings-v5-text-nano", "device": "cpu", "vram_gb": 0, "multimodal": false}
```

## OpenAI (+ Voyage AI)

```bash
curl -s http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["Hello world"], "model": "jina-embeddings-v5-text-nano"}'
```

Optional fields:

- `task` - one of the model's supported tasks (see [task list](#embedding-tasks))
- `dimensions` - truncate to any supported matryoshka dim (e.g. 128, 256, 512)
- `input_type` (Voyage) - `query` / `document`, mapped to the model's `task`
- `output_dimension` (Voyage) - alias for `dimensions`

Python:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
resp = client.embeddings.create(model="jina-embeddings-v5-text-nano", input=["Hello world"])
```

## Cohere

```bash
curl -s http://localhost:8080/v1/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["Hello world"], "model": "jina-v5-nano", "input_type": "search_query"}'
```

`input_type` maps to the model's task: `search_query` -> retrieval (query side), `search_document` -> retrieval (passage side), `classification`, `clustering`.

## Gemini

```bash
curl -s "http://localhost:8080/v1/models/jina-embeddings-v5-text-nano:embedContent" \
  -H 'Content-Type: application/json' \
  -d '{"content": {"parts": [{"text": "Hello world"}]}, "taskType": "RETRIEVAL_QUERY"}'
```

Batch: `:batchEmbedContents` with `{"requests": [...]}`.

## Voyage AI Multimodal

```bash
curl -s http://localhost:8080/v1/multimodalembeddings \
  -H 'Content-Type: application/json' \
  -d '{"inputs": [{"content": [{"type": "text", "text": "Hello"}]}], "model": "voyage-multimodal-3.5"}'
```

## Reranker models

```bash
curl -s http://localhost:8080/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "best embedding model",
    "documents": ["jina-v5 is multilingual", "weather is sunny", "embeddings represent semantic meaning"],
    "top_n": 2
  }'
```

The reranker endpoint follows Cohere's `/v1/rerank` shape: `query`, `documents`, optional `top_n`, returns `{"results": [{"index": i, "relevance_score": s}, ...]}`.

## Multimodal inputs (omni / clip / vlm)

`v5-omni-small`, `v5-omni-nano`, `v4`, `jina-clip-v2`, and `jina-vlm` accept images, audio, and video alongside text. Media must be base64-encoded (no URLs - the air-gap guarantee forbids outbound fetches). Max 10 MB per input.

```bash
# Single image
curl -s http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": [{"type": "image_base64", "image_base64": {"base64": "<B64>", "mime_type": "image/png"}}]}'

# Fused text + image -> single embedding
curl -s http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": [{"content": [{"type": "text", "text": "A red square"}, {"type": "image", "format": "base64", "value": "<B64>"}]}]}'
```

Text-only models return HTTP 400 if multimodal inputs are sent.

## Embedding tasks

Pass `task` to optimize embeddings for your use case. The server maps Cohere's `input_type` and Gemini's `taskType` automatically.

| Model family | Supported tasks |
|---|---|
| v5-text / v5-omni | `retrieval` (default), `text-matching`, `classification`, `clustering` |
| v4 | `retrieval` (default), `text-matching`, `code` |
| v3 | `retrieval.query`, `retrieval.passage`, `text-matching`, `classification`, `separation` |
| v2 / v1 | (no task field; passing one is ignored) |

## Elasticsearch integration

Drop-in for ES inference service:

```json
PUT _inference/text_embedding/jina-local
{
  "service": "openai",
  "service_settings": {
    "url": "http://your-host:8080/v1/embeddings",
    "model_id": "jina-embeddings-v5-text-nano",
    "api_key": "not-needed"
  }
}
```

Then index and query as usual via the `inference_id`. The `api_key` is required by the ES schema but unused server-side.

## Source of truth

The full request/response shapes live in [`server/app.py`](https://github.com/jina-ai/jina-airgap/blob/main/server/app.py). Run the server, hit `GET /docs` for the auto-generated Swagger UI.
