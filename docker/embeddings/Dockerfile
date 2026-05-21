# Jina AI Embedding Model - Air-Gapped Docker Image
# Multi-stage: stage 1 downloads weights, stage 2 is the lean runtime

ARG MODEL_ID=jinaai/jina-embeddings-v3
ARG HF_TOKEN=""

# Stage 1: Download model weights
FROM python:3.11-slim AS downloader

ARG MODEL_ID
ARG HF_TOKEN

RUN pip install --no-cache-dir huggingface_hub

RUN mkdir -p /model_cache
ENV HF_HOME=/model_cache

RUN python -c "\
from huggingface_hub import snapshot_download; \
import os; \
token = os.environ.get('HF_TOKEN') or None; \
snapshot_download('${MODEL_ID}', local_dir='/model_cache/model', token=token, ignore_patterns=['*.ot', '*.msgpack', 'flax_model*', 'tf_model*', 'rust_model*'])"


# Stage 2: Runtime image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Copy server code
COPY server/ /app/

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy model weights from downloader stage
COPY --from=downloader /model_cache/model /model

# Enforce air-gap
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HOME=/model_cache
ENV JINA_MODEL_TYPE=embedding

# Model ID is the local path to baked-in weights
ENV JINA_MODEL_ID=/model

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "app.py"]
