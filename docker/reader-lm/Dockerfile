# Jina AI Reader LM - Air-Gapped Docker Image

ARG MODEL_ID=jinaai/ReaderLM-v2
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

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

COPY server/ /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY --from=downloader /model_cache/model /model

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV JINA_OFFLINE=1
ENV HF_HOME=/model_cache
ENV JINA_MODEL_TYPE=reader
ENV JINA_MODEL_ID=/model

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "app.py"]
