#!/usr/bin/env bash
# Pull a prebuilt jina-airgap image from GHCR and save as .tar.gz for offline transport.
# Skip the bundle phase entirely when a prebuilt exists.
#
# Usage:
#   ./scripts/pull-prebuilt.sh MODEL [RUNTIME]
#
# Examples:
#   ./scripts/pull-prebuilt.sh jina-embeddings-v5-text-nano        # default: cpu
#   ./scripts/pull-prebuilt.sh jina-embeddings-v5-text-small gpu
#
# Output: MODEL-RUNTIME.tar.gz in the current directory.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 MODEL [cpu|gpu]" >&2
  echo
  echo "List available prebuilt models in the README's 'Prebuilt' column." >&2
  exit 1
fi

MODEL="$1"
RUNTIME="${2:-cpu}"

if [[ "$RUNTIME" != "cpu" && "$RUNTIME" != "gpu" ]]; then
  echo "error: RUNTIME must be 'cpu' or 'gpu' (got: $RUNTIME)" >&2
  exit 1
fi

REGISTRY="ghcr.io/jina-ai/jina-airgap"
SRC="${REGISTRY}/${MODEL}:${RUNTIME}"
LOCAL_TAG="jina/${MODEL}:${RUNTIME}"
OUTPUT="${MODEL}-${RUNTIME}.tar.gz"

echo "Pulling $SRC ..."
if ! docker pull "$SRC" 2>&1 | tee /tmp/pull-prebuilt.log; then
  if grep -q "unauthorized\|denied" /tmp/pull-prebuilt.log; then
    cat >&2 <<EOF

Pull failed with unauthorized/denied. GHCR requires authentication.

Login first:
  echo YOUR_GH_TOKEN | docker login ghcr.io -u YOUR_GH_USERNAME --password-stdin

Create a token at https://github.com/settings/tokens/new?scopes=read:packages

If you use sudo, run docker login as both user and root (separate credential stores).
EOF
  fi
  rm -f /tmp/pull-prebuilt.log
  exit 1
fi
rm -f /tmp/pull-prebuilt.log

echo "Retagging as $LOCAL_TAG ..."
docker tag "$SRC" "$LOCAL_TAG"

echo "Saving to $OUTPUT ..."
docker save "$LOCAL_TAG" | gzip > "$OUTPUT"

SIZE=$(du -h "$OUTPUT" | cut -f1)
echo
echo "Done. $OUTPUT ($SIZE)"
echo
echo "Transfer this file to the air-gapped machine, then:"
echo "  docker load < $OUTPUT"
echo "  docker run -p 8080:8080 $LOCAL_TAG"
[[ "$RUNTIME" == "gpu" ]] && echo "  # for GPU runtime: docker run --gpus all -p 8080:8080 $LOCAL_TAG"
