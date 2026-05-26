# Bundling Guide

When to bundle from scratch instead of pulling a prebuilt:

- A model you want isn't in the [prebuilt list](Model-Catalog) yet
- You want to pin different dependency versions
- You're operating under a policy that forbids `docker pull` from third-party registries

The bundle phase needs network access and ~30GB free disk per model. Output is a single `.tar.gz`.

![bundle](images/02-bundle.gif)

The CLI also includes a [`list` command](#cli-commands) for browsing the catalog:

![list](images/01-list.gif)

## Prerequisites (connected machine)

- Linux (Ubuntu 22.04 tested), x86_64
- Docker 24+ with BuildKit
- Python 3.10+ (the CLI is a single-file script with no third-party imports)
- 30GB+ free disk (200GB+ if building multiple)
- For GPU bundles: no GPU is required at build time (the Dockerfile uses the PyTorch CUDA base image), but the runtime host that deploys the image will need an L4-class GPU or better.

## Bundle on GCP L4

The repo includes [`scripts/bootstrap-gcp.sh`](https://github.com/jina-ai/jina-airgap/blob/main/scripts/bootstrap-gcp.sh), a one-shot provisioner. It creates a `g2-standard-4` + L4 instance with Docker + NVIDIA Container Toolkit + the repo pre-cloned:

```bash
git clone https://github.com/jina-ai/jina-airgap.git
cd jina-airgap
./scripts/bootstrap-gcp.sh                     # defaults: us-central1-a, g2-standard-4, 1xL4
# or CPU-only builder (cheaper, fine for CPU bundles):
GPU_COUNT=0 MACHINE_TYPE=e2-standard-4 ./scripts/bootstrap-gcp.sh my-builder us-central1-a
```

> **L4 stockouts are common in popular US zones.** If the script errors with "does not have enough resources", retry in `us-west1-a`, `us-east4-a`, `europe-west4-a`, or `asia-southeast1-a`. See [Troubleshooting](Troubleshooting#l4-stockout).

When provisioning finishes you'll get an SSH command. From inside:

```bash
cd ~/jina-airgap
sg docker -c 'python3 jina-airgap.py bundle --model jina-embeddings-v5-text-nano --cpu-only --yes'
```

`sg docker -c '...'` is only needed in the same session that just installed Docker. After a reconnect, `docker` works without it.

## CLI commands

```bash
python3 jina-airgap.py list                               # show all models
python3 jina-airgap.py list --type embedding --verbose    # filter + extras
python3 jina-airgap.py list --json                        # machine-readable

python3 jina-airgap.py bundle                             # interactive picker
python3 jina-airgap.py bundle --model MODEL               # GPU runtime, named model
python3 jina-airgap.py bundle --model MODEL --cpu-only    # CPU-only image
python3 jina-airgap.py bundle --model MODEL --yes         # non-interactive (CI)

python3 jina-airgap.py deploy --image PATH.tar.gz         # docker load + run (testing)
python3 jina-airgap.py serve --model MODEL                # run directly without Docker
```

Backward-compatible aliases (`pack` -> `bundle`, `load` -> `deploy`) still work.

## Output

A successful bundle produces:

```
jina-MODEL-cpu.tar.gz        # for --cpu-only
jina-MODEL-gpu.tar.gz        # for GPU runtime
```

Sizes: 2-4GB for nano/small text models, up to ~12GB for omni/v4.

## CPU vs GPU runtime

|  | CPU image | GPU image |
|---|---|---|
| Base image | `python:3.11-slim` | `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel` |
| Image size | smaller (~2-4GB) | larger (~8-12GB) |
| Runs on host with GPU | yes (ignores GPU) | yes |
| Runs on host without GPU | yes | no |
| Inference speed | ~10x slower | full speed |
| Build wall-clock | ~5-15 min | ~10-25 min |

If unsure, build both. The runtime host decides which to load.

## Disk hygiene during multi-model builds

Each bundle holds the full image + the tarball + build cache. Reclaim between bundles:

```bash
docker builder prune -af      # reclaim BuildKit cache
docker system prune -f        # remove dangling images
rm jina-OLDMODEL-*.tar.gz     # remove tars you've already transferred out
```

CONTRIBUTING.md documents [batch-building 6 priority models](https://github.com/jina-ai/jina-airgap/blob/main/CONTRIBUTING.md#batch-build-all-6-priority-models) with this pattern.

## Transfer to the air-gapped machine

The `.tar.gz` is a single self-contained file. Move it however your policy allows: SCP, SFTP, USB drive, S3 bucket the air-gapped network can reach, sneakernet to a removable disk. On the target host:

```bash
docker load < jina-MODEL-cpu.tar.gz
docker run -d -p 8080:8080 jina/MODEL:cpu
curl http://localhost:8080/health
```

That's the entire air-gapped deploy step. The image has `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` baked in - any code path that would call out to HuggingFace at runtime fails immediately rather than silently downloading.

## Caveats worth knowing before you build

- Each model's `transformers` version is pinned. See [Troubleshooting -> Transformers version pins](Troubleshooting#transformers-version-pins) for why.
- Reranker models load as `CrossEncoder`, not `SentenceTransformer`. Already handled in the server, but relevant if you `serve` directly.
- v5-omni models need ~30GB disk during the build (large base image + flash-attn compile). On 100GB hosts you can hold ~4 omni bundles at once.

Full debug history lives in [CONTRIBUTING.md "Known Caveats"](https://github.com/jina-ai/jina-airgap/blob/main/CONTRIBUTING.md#known-caveats-from-hard-won-debugging).
