# Troubleshooting

If you hit something not listed here, please [open an issue](https://github.com/jina-ai/jina-airgap/issues/new) with the full error and `docker logs <container>` output.

## Docker permission denied

**Symptom**

```
permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```

or from the CLI:

```
Error: Docker is not installed or not running.
```

**Cause**: your user isn't in the `docker` group, or the group membership hasn't been applied to your current shell session.

**Fix**:

```bash
sudo usermod -aG docker $USER
# then either reconnect SSH / open a new terminal, OR:
sg docker -c 'docker ps'        # apply group in current session
# OR prefix one-off commands with sudo:
sudo docker ps
```

## L4 stockout

**Symptom** (from `gcloud compute instances create`):

```
The zone 'projects/PROJECT/zones/ZONE' does not have enough resources to fulfill the request.
'NULL:0/NULL:0/NULL:0 (state:STOCKOUT, sub-state:STOCKOUT, resource type:compute)'
```

**Cause**: L4 GPUs are extremely popular in US zones. As of May 2026 all `us-central1-*`, `us-west1-*`, `us-east1-*`, `us-east4-*` were simultaneously out.

**Fix**: try zones in this order, picking the first that succeeds:

1. `us-west1-a`, `us-east4-a` (cheapest egress to most clients)
2. `europe-west4-a`, `europe-west1-b`
3. `asia-southeast1-a` (Singapore - usually available)
4. As a last resort, switch to A100 or T4 in a US zone with quota.

A retry loop is included at the bottom of [`scripts/bootstrap-gcp.sh`](https://github.com/jina-ai/jina-airgap/blob/main/scripts/bootstrap-gcp.sh).

## CUDA mismatch

**Symptom**: GPU container exits immediately with `CUDA error: no kernel image is available for execution on the device` or `forward compatibility was attempted on non supported HW`.

**Cause**: the host's NVIDIA driver is older than what the image was compiled against. The GPU image targets CUDA 12.1; you need driver `>=525.60`.

**Fix**:

```bash
nvidia-smi   # check the Driver Version line, NOT the CUDA version line
```

If `<525`, either update the driver, or use the CPU image (`:cpu` tag) which has no CUDA dependency.

## Out of memory on GPU

**Symptom**: `CUDA out of memory. Tried to allocate XX MiB`.

**Cause**: model is too big for the GPU's VRAM, or another container is already using it.

**Fix**: check VRAM with `nvidia-smi`. The [Model Catalog](Model-Catalog) lists per-model VRAM. If you have 24GB and the model says ~8GB, you should fit two replicas but not four. Reduce batch size at the client (smaller `input` arrays) or pick a smaller model.

## Transformers version pins

**Symptom**: `ImportError: cannot import name 'Qwen3Config' from 'transformers'` or similar `cannot import` errors when running `serve` directly (not via Docker).

**Cause**: model needs a specific transformers version. Each model's `deps` block in `models/catalog.json` pins it; the Docker image installs exactly that version. If you `serve` outside Docker you must mirror the pins.

| Model family | transformers | Why |
|---|---|---|
| v5-text-nano, v5-text-small | `==4.51.0` | needs `Qwen3Config` (added in 4.51) |
| v5-omni-nano, v5-omni-small | `==4.57.0` | needs `Qwen3VLVisionConfig` (added in 4.57) |
| v3 | `==4.48.3` | older base, no qwen3 dependency |
| reranker-v3 | `==4.51.0` | based on Qwen3 |

> The bundle phase deletes each model repo's own `requirements.txt` after download to prevent runtime auto-upgrade by `trust_remote_code` paths.

## Reranker returns 400 on `/v1/embeddings`

Reranker models don't expose `/v1/embeddings`. Use `/v1/rerank` instead. See [API Reference -> Reranker models](API-Reference#reranker-models).

## Empty / NaN embeddings on Qwen3 reranker

**Symptom**: reranker scores are all 0.5 or all NaN.

**Cause**: Qwen3-based rerankers need `pad_token = eos_token`. The server sets this automatically. If you're loading the model with the SentenceTransformers SDK directly:

```python
model = CrossEncoder(...)  # NOT SentenceTransformer
model.tokenizer.pad_token = model.tokenizer.eos_token
```

## Don't use `--network=none` for "air-gap testing"

**Symptom**: testing the image with `docker run --network=none ...` and the container can be reached only from inside the container.

**Cause**: `--network=none` removes the entire network stack, including localhost-binding. The host can't reach `localhost:8080` either, because port forwarding doesn't work without a network namespace.

**Fix**: don't use `--network=none`. The real air-gap guarantee is that `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are baked into the image, so any code path that would download a weight fails immediately. Run normally with `-p 8080:8080`.

## Bundle build fails on `flash-attn` compile

**Symptom**: GPU bundle build hangs or OOMs during the `pip install flash-attn` step.

**Cause**: `flash-attn` requires `nvcc` (CUDA compiler) which isn't in the runtime-only base image. The Dockerfile uses the `-devel` variant of pytorch/pytorch which includes nvcc - if you've forked the Dockerfile, make sure you're still on `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`.

## Spot instance preempted mid-build

**Symptom**: SSH disconnects, instance shows `TERMINATED` in `gcloud compute instances list`.

**Cause**: GCP spot/preemptible instances can be reclaimed any time.

**Fix**: start bundles with `nohup` so disconnection doesn't kill the build:

```bash
nohup bash -c 'cd ~/jina-airgap && python3 jina-airgap.py bundle --model X --yes' > ~/bundle.log 2>&1 &
disown
```

Reconnect later and `tail -f ~/bundle.log` to check progress. Docker images persist across restart; `/tmp` does not, so write logs to `~/`.

## Disk full during multi-model build

**Symptom**: `no space left on device` after building two or three bundles.

**Cause**: each bundle leaves the BuildKit cache, the built image, and the `.tar.gz` on disk - cumulatively ~10-20GB per model.

**Fix**:

```bash
docker builder prune -af       # reclaim BuildKit cache (safe)
docker image prune -f          # dangling intermediate images
rm jina-OLDMODEL-*.tar.gz      # tarballs you've already transferred out
docker system df               # see what's left
```

## OCI labels missing on pushed images

**Symptom**: image pushed to GHCR but doesn't appear in the repo's package list.

**Cause**: the `LABEL org.opencontainers.image.source=https://github.com/jina-ai/jina-airgap` is missing from the Dockerfile.

**Fix**: both `docker/Dockerfile.cpu` and `docker/Dockerfile.gpu` already include this label. If you've forked them, keep the LABEL. After fixing, retag and push - GitHub picks up the link on next push.
