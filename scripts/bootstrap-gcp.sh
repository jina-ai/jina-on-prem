#!/usr/bin/env bash
# Provision a GCP instance ready to run `python jina-airgap.py bundle`.
#
# Usage:
#   ./scripts/bootstrap-gcp.sh [INSTANCE_NAME] [ZONE]
#
# Defaults: jina-airgap-builder, us-central1-a.
# Override with env vars: PROJECT, MACHINE_TYPE, GPU_TYPE, GPU_COUNT, DISK_GB.
# Set GPU_COUNT=0 to provision a CPU-only builder.

set -euo pipefail

NAME="${1:-jina-airgap-builder}"
ZONE="${2:-us-central1-a}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-4}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
GPU_COUNT="${GPU_COUNT:-1}"
DISK_GB="${DISK_GB:-200}"

if [[ -z "$PROJECT" ]]; then
  echo "error: no PROJECT set. run 'gcloud config set project YOUR_PROJECT' or pass PROJECT=..." >&2
  exit 1
fi

gpu_args=()
if [[ "$GPU_COUNT" -gt 0 ]]; then
  gpu_args=(--accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" --maintenance-policy=TERMINATE)
  IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
else
  IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2204-lts}"
fi

IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
[[ "$GPU_COUNT" -eq 0 ]] && IMAGE_PROJECT="${IMAGE_PROJECT_CPU:-ubuntu-os-cloud}"

echo "Creating $NAME in $ZONE (project=$PROJECT, $MACHINE_TYPE, ${GPU_COUNT}x${GPU_TYPE}, ${DISK_GB}GB)..."

gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="${DISK_GB}GB" \
  --boot-disk-type=pd-balanced \
  --scopes=cloud-platform \
  --metadata="install-nvidia-driver=True" \
  --labels="purpose=jina-airgap-builder" \
  "${gpu_args[@]}"

echo
echo "Waiting 30s for instance boot..."
sleep 30

echo "Installing Docker and NVIDIA Container Toolkit..."
gcloud compute ssh "$NAME" --zone="$ZONE" --project="$PROJECT" --command='
set -e
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh > /tmp/docker-install.log 2>&1
fi
sudo usermod -aG docker $(whoami)

if command -v nvidia-smi >/dev/null 2>&1 && ! dpkg -l | grep -q nvidia-container-toolkit; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o /tmp/nvgpg.key
  sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg /tmp/nvgpg.key
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list -o /tmp/nct.list
  sudo sed -i "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" /tmp/nct.list
  sudo cp /tmp/nct.list /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update -qq
  sudo apt-get install -y -qq nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker > /dev/null
  sudo systemctl restart docker
fi

if [ ! -d ~/jina-airgap ]; then
  git clone https://github.com/jina-ai/jina-airgap.git
fi
'

cat <<EOF

Done. SSH in with:
  gcloud compute ssh $NAME --zone=$ZONE --project=$PROJECT

Then build a bundle:
  cd ~/jina-airgap
  sg docker -c 'python3 jina-airgap.py bundle --model jina-embeddings-v5-text-nano --cpu-only --yes'

When finished, delete the instance:
  gcloud compute instances delete $NAME --zone=$ZONE --project=$PROJECT --quiet
EOF
