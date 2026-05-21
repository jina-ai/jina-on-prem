# Jina AI Embedding Model - Air-Gapped Docker Image
# Bakes model weights + dynamic modules into the image so it runs fully offline.

ARG MODEL_ID=jinaai/jina-embeddings-v3
ARG HF_TOKEN=""

# Stage 1: Download model weights + pre-cache dynamic modules
FROM python:3.11-slim AS downloader

ARG MODEL_ID
ARG HF_TOKEN

ENV HF_HOME=/model_cache
ENV HF_TOKEN=${HF_TOKEN}

RUN pip install --no-cache-dir huggingface_hub sentence-transformers transformers torch --index-url https://download.pytorch.org/whl/cpu

# Download weights into HF cache structure and pre-populate dynamic module cache
RUN python -c "
import os
os.environ['HF_HOME'] = '/model_cache'
token = os.environ.get('HF_TOKEN') or None
model_id = '${MODEL_ID}'
print(f'Downloading {model_id}...')

from huggingface_hub import snapshot_download
snapshot_download(model_id, token=token, ignore_patterns=['*.ot', '*.msgpack', 'flax_model*', 'tf_model*', 'rust_model*'])

# Pre-load to cache trust_remote_code dynamic modules
from sentence_transformers import SentenceTransformer
SentenceTransformer(model_id, trust_remote_code=True, device='cpu')
print('Done caching model and modules')

# Write model ID for runtime
with open('/model_cache/MODEL_ID', 'w') as f:
    f.write(model_id)
"


# Stage 2: Runtime image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-dev curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Copy server code
COPY server/ /app/

# Install Python deps (CUDA-enabled torch)
RUN pip install --no-cache-dir -r requirements.txt

# Copy cached model and dynamic modules from downloader stage
COPY --from=downloader /model_cache /model_cache

# Enforce air-gap: all model files are baked in
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV JINA_OFFLINE=1
ENV HF_HOME=/model_cache
ENV JINA_MODEL_TYPE=embedding

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -sf http://localhost:8080/health || exit 1

# Entrypoint reads model ID from baked-in file
CMD ["sh", "-c", "JINA_MODEL_ID=$(cat /model_cache/MODEL_ID) exec python app.py"]
