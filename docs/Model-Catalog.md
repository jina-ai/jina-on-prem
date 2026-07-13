All 28 models supported by jina-on-prem. Auto-generated from [`models/catalog.json`](https://github.com/jina-ai/jina-on-prem/blob/main/models/catalog.json) - re-run `python3 scripts/gen_catalog_md.py` to refresh.

**License note**: Models tagged `CC-BY-NC-4.0` need a commercial license for production use. Contact [Elastic sales](https://www.elastic.co/contact).

## Embeddings

| Model | Prebuilt | Params | VRAM | Context | Output | Modality | License |
|---|---|---|---|---|---|---|---|
| `jina-embeddings-v5-omni-small` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v5-omni-small/894209247) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v5-omni-small/894209342) | 1.74B | ~8GB | 32K | 1024 (matryoshka: 32-1024) | multimodal | CC-BY-NC-4.0 |
| `jina-embeddings-v5-omni-nano` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v5-omni-nano/894208735) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v5-omni-nano/894209314) | 1.04B | ~5GB | 8K | 768 (matryoshka: 32-768) | multimodal | CC-BY-NC-4.0 |
| `jina-embeddings-v5-text-small` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v5-text-small/886085110) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v5-text-small/886086903) | 677M | ~3GB | 32K | 1024 (matryoshka: 32-1024) | text | CC-BY-NC-4.0 |
| `jina-embeddings-v5-text-nano` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v5-text-nano/886084948) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v5-text-nano/886086707) | 239M | ~2GB | 8K | 768 (matryoshka: 32-768) | text | CC-BY-NC-4.0 |
| `jina-code-embeddings-1.5b` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-code-embeddings-1.5b/887526855) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-code-embeddings-1.5b/887526880) | 1.5B | ~4GB | 32K | 1536 (matryoshka: 128-1536) | code | CC-BY-NC-4.0 |
| `jina-code-embeddings-0.5b` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-code-embeddings-0.5b/887504159) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-code-embeddings-0.5b/887504265) | 494M | ~2GB | 32K | 896 (matryoshka: 64-896) | code | CC-BY-NC-4.0 |
| `jina-embeddings-v4` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v4/887579512) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v4/887591515) | 3.8B | ~10GB | 32K | 2048 (matryoshka: 128-2048) | multimodal | Qwen Research License |
| `jina-clip-v2` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-clip-v2/886502690) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-clip-v2/886416640) | 865M | ~4GB | 8K | 1024 (matryoshka: 64-1024) | multimodal | CC-BY-NC-4.0 |
| `jina-embeddings-v3` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v3/893876768) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v3/893876917) | 570M | ~3GB | 8K | 1024 (matryoshka: 32-1024) | text | CC-BY-NC-4.0 |
| `jina-clip-v1` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-clip-v1/886557142) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-clip-v1/886557272) | 223M | ~1GB | 8K | 768 | multimodal | Apache-2.0 |
| `jina-embeddings-v2-base-es` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v2-base-es/886918894) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v2-base-es/886937006) | 161M | ~1GB | 8K | 768 | text | Apache-2.0 |
| `jina-embeddings-v2-base-code` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v2-base-code/886944161) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v2-base-code/886961431) | 137M | ~1GB | 8K | 768 | code | Apache-2.0 |
| `jina-embeddings-v2-base-de` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v2-base-de/886902047) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v2-base-de/886912798) | 161M | ~1GB | 8K | 768 | text | Apache-2.0 |
| `jina-embeddings-v2-base-zh` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v2-base-zh/886883815) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v2-base-zh/886895428) | 161M | ~1GB | 8K | 768 | text | Apache-2.0 |
| `jina-embeddings-v2-base-en` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v2-base-en/886838080) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embeddings-v2-base-en/886838386) | 137M | ~1GB | 8K | 768 | text | Apache-2.0 |
| `jina-embedding-b-en-v1` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embedding-b-en-v1/886967319) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-embedding-b-en-v1/886983429) | 110M | ~1GB | 512 | 768 | text | Apache-2.0 |

## Rerankers

| Model | Prebuilt | Params | VRAM | Context | Output | Modality | License |
|---|---|---|---|---|---|---|---|
| `jina-reranker-v3` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-reranker-v3/893272991) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-reranker-v3/893273806) | 597M | ~3GB | 131K | - | text | CC-BY-NC-4.0 |
| `jina-reranker-m0` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-reranker-m0/887687921) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-reranker-m0/887688419) | 2.4B | ~6GB | 10K | - | multimodal | CC-BY-NC-4.0 |
| `jina-reranker-v2-base-multilingual` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-reranker-v2-base-multilingual/887650292) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-reranker-v2-base-multilingual/887650459) | 278M | ~1GB | 1K | - | text | CC-BY-NC-4.0 |
| `jina-reranker-v1-turbo-en` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-reranker-v1-turbo-en/893039415) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-reranker-v1-turbo-en/893160841) | 37.8M | ~1GB | 8K | - | text | Apache-2.0 |
| `jina-reranker-v1-tiny-en` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-reranker-v1-tiny-en/893020925) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-reranker-v1-tiny-en/893105290) | 33M | ~1GB | 8K | - | text | Apache-2.0 |
| `jina-reranker-v1-base-en` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-reranker-v1-base-en/893052182) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-reranker-v1-base-en/893209013) | 137M | ~1GB | 8K | - | text | Apache-2.0 |

## ColBERT

| Model | Prebuilt | Params | VRAM | Context | Output | Modality | License |
|---|---|---|---|---|---|---|---|
| `jina-colbert-v2` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-colbert-v2/887811941) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-colbert-v2/887811981) | 560M | ~3GB | 8K | 128 (matryoshka: 64-128) | text | CC-BY-NC-4.0 |
| `jina-colbert-v1-en` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-colbert-v1-en/887789554) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-colbert-v1-en/887790076) | 137M | ~1GB | 8K | 128 | text | Apache-2.0 |

## Readers

| Model | Prebuilt | Params | VRAM | Context | Output | Modality | License |
|---|---|---|---|---|---|---|---|
| `ReaderLM-v2` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2FReaderLM-v2/889912322) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2FReaderLM-v2/889913328) | 1.54B | ~4GB | 524K | - | text | CC-BY-NC-4.0 |
| `reader-lm-1.5b` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Freader-lm-1.5b/889536644) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Freader-lm-1.5b/889538436) | 1.54B | ~4GB | 262K | - | text | CC-BY-NC-4.0 |
| `reader-lm-0.5b` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Freader-lm-0.5b/889498632) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Freader-lm-0.5b/889499035) | 494M | ~2GB | 262K | - | text | CC-BY-NC-4.0 |

## Vision-Language

| Model | Prebuilt | Params | VRAM | Context | Output | Modality | License |
|---|---|---|---|---|---|---|---|
| `jina-vlm` | [cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-vlm/886722811) / [gpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2Fjina-vlm/886722952) | 2.4B | ~6GB | 32K | - | multimodal | CC-BY-NC-4.0 |

## Picking a model

Quick rules of thumb:

- **First-time test / latency-critical**: `jina-embeddings-v5-text-nano` (239M, ~2GB, CPU-friendly).
- **Multilingual production embeddings**: `jina-embeddings-v5-text-small` or `jina-embeddings-v4`.
- **Multimodal (text + image)**: `jina-embeddings-v5-omni-small` or `jina-clip-v2`.
- **Code search**: `jina-code-embeddings-1.5b` (or 0.5b for smaller deploys).
- **Reranking after retrieval**: `jina-reranker-v3` (best quality) or `jina-reranker-v2-base-multilingual` (faster).
- **HTML/document cleanup**: `ReaderLM-v2` (largest context) or `reader-lm-0.5b` (lightweight).

See [API Reference](API-Reference) for the request shapes each model expects.
