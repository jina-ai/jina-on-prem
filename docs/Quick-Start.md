# Quick Start

Goal: get your first `/v1/embeddings` response in under five minutes, on a machine that has Docker.

We'll use a prebuilt image (`jina-embeddings-v5-text-nano`, CPU). No GPU needed. No Python needed.

![quick-start](images/03-deploy.gif)

## Prerequisites

- Docker installed and running (`docker ps` should work without error)
- ~3GB free disk for the image + `.tar.gz`
- Port 8080 available

> **Air-gapped target machine?** Do steps 1-3 on a connected machine, transfer the `.tar.gz`, and resume from step 4 offline.

## 1. Pull a prebuilt image

```bash
docker pull ghcr.io/jina-ai/jina-airgap/jina-embeddings-v5-text-nano:cpu
docker tag ghcr.io/jina-ai/jina-airgap/jina-embeddings-v5-text-nano:cpu \
           jina/jina-embeddings-v5-text-nano:cpu
```

(The retag is optional but matches the names used in examples below.)

## 2. (Optional) Export for offline transport

If your target machine has no network:

```bash
docker save jina/jina-embeddings-v5-text-nano:cpu | gzip > jina-v5-nano.tar.gz
# Transfer jina-v5-nano.tar.gz to the air-gapped machine, then:
docker load < jina-v5-nano.tar.gz
```

The repo ships [`scripts/pull-prebuilt.sh`](https://github.com/jina-ai/jina-airgap/blob/main/scripts/pull-prebuilt.sh) which does both steps:

```bash
./scripts/pull-prebuilt.sh jina-embeddings-v5-text-nano cpu
```

## 3. Run it

```bash
docker run -d --name jina-nano -p 8080:8080 jina/jina-embeddings-v5-text-nano:cpu
docker logs -f jina-nano   # watch for "Uvicorn running on http://0.0.0.0:8080"
```

For GPU: add `--gpus all` and use the `:gpu` tag.

## 4. Verify

```bash
curl http://localhost:8080/health
# {"status":"ok","model":"jinaai/jina-embeddings-v5-text-nano","device":"cpu","ready":true,"multimodal":false,"schemas":["openai","voyage","gemini","cohere"]}

curl -s http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["Hello world"],"model":"jina-embeddings-v5-text-nano"}' \
  | jq '.data[0].embedding | length'
# 768
```

If both calls succeed you're done. From here:

- Plug the URL into your app. The server speaks four API schemas - see [API Reference](API-Reference).
- Pick a different model from the [Model Catalog](Model-Catalog).
- Build your own bundle from scratch via the [Bundling Guide](Bundling-Guide).

## Common first-time errors

- `permission denied while trying to connect to the docker API`: you're not in the `docker` group. Either `sudo usermod -aG docker $USER` then reconnect, or prefix commands with `sudo`. See [Troubleshooting](Troubleshooting#docker-permission-denied).
- `bind: address already in use`: something is on port 8080. Map a different port: `-p 9090:8080`.
- Container exits immediately on GPU run: check `docker logs <container>` for CUDA mismatch. See [Troubleshooting](Troubleshooting#cuda-mismatch).
