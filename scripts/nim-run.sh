#!/usr/bin/env bash
# Run a Jina AI model via NVIDIA NIM (model-free-nim).
#
# Usage:
#   ./scripts/nim-run.sh <model-id> [port] [extra-docker-args...]
#
# Examples:
#   ./scripts/nim-run.sh jina-embeddings-v3
#   ./scripts/nim-run.sh jina-reranker-v3 8001
#   ./scripts/nim-run.sh ReaderLM-v2 8002
#
# Environment:
#   NGC_API_KEY  - Required. Your NVIDIA NGC API key (nvapi-... format).
#   NIM_CACHE    - Optional. Host path for model weight cache. Default: $HOME/nim_cache
#   NIM_IMAGE    - Optional. Override the NIM image. Default: nvcr.io/nim/nvidia/model-free-nim:2.0.6
#
# Model IDs correspond to HuggingFace repo names under jinaai/:
#   jina-embeddings-v3, jina-embeddings-v5-text-small,
#   jina-reranker-v3, ReaderLM-v2
#
# Endpoints after startup:
#   Embeddings:   POST http://localhost:<port>/v1/embeddings
#   Reranking:    POST http://localhost:<port>/rerank     (NOT /v1/rerank)
#   Chat/LM:      POST http://localhost:<port>/v1/chat/completions
#   Health:       GET  http://localhost:<port>/v1/health/ready

set -euo pipefail

MODEL_ID="${1:-}"
PORT="${2:-8000}"
NIM_IMAGE="${NIM_IMAGE:-nvcr.io/nim/nvidia/model-free-nim:2.0.6}"
NIM_CACHE="${NIM_CACHE:-$HOME/nim_cache}"

if [[ -z "$MODEL_ID" ]]; then
  echo "Usage: $0 <model-id> [port]"
  echo ""
  echo "Supported models (NIM-tested):"
  echo "  jina-embeddings-v3"
  echo "  jina-embeddings-v5-text-small"
  echo "  jina-reranker-v3"
  echo "  ReaderLM-v2"
  exit 1
fi

if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "Error: NGC_API_KEY is not set."
  echo "Get a free key at https://ngc.nvidia.com/setup"
  exit 1
fi

HF_URI="hf://jinaai/${MODEL_ID}"
CONTAINER_NAME="jina-nim-${MODEL_ID//\//-}"

echo "Starting ${MODEL_ID} via NIM..."
echo "  Image:     ${NIM_IMAGE}"
echo "  Model:     ${HF_URI}"
echo "  Port:      ${PORT} -> 8000"
echo "  Cache:     ${NIM_CACHE}"
echo "  Container: ${CONTAINER_NAME}"
echo ""

# Remove stale container with the same name if it exists
if docker inspect "${CONTAINER_NAME}" &>/dev/null; then
  echo "Removing existing container ${CONTAINER_NAME}..."
  docker rm -f "${CONTAINER_NAME}"
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --shm-size=16GB \
  -p "${PORT}:8000" \
  -v "${NIM_CACHE}:/opt/nim/.cache" \
  -e NGC_API_KEY="${NGC_API_KEY}" \
  "${NIM_IMAGE}" \
  "${HF_URI}" --trust-remote-code

echo ""
echo "Container started. Waiting for ready..."
echo "(First run downloads weights — may take a few minutes.)"
echo ""

until curl -sf "http://localhost:${PORT}/v1/health/ready" >/dev/null 2>&1; do
  printf "."
  sleep 5
done

echo ""
echo "Ready! ${MODEL_ID} is serving on port ${PORT}."
echo ""
echo "Quick test:"
echo "  curl -X POST http://localhost:${PORT}/v1/embeddings \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"ga-model-free-nim\", \"input\": \"hello world\"}'"
