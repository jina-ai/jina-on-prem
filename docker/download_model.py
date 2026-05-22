"""
Model download script - runs inside the Docker build downloader stage.
Downloads weights AND all model code files into HF cache structure.
At runtime, HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 ensures models load
from this local cache without any network calls.
"""
import os
import sys

os.environ["HF_HOME"] = "/model_cache"
token = os.environ.get("HF_TOKEN") or None
model_id = os.environ.get("MODEL_ID", "")
if not model_id:
    print("ERROR: MODEL_ID env var not set", file=sys.stderr)
    sys.exit(1)

print(f"Downloading {model_id}...")

from huggingface_hub import snapshot_download

# Download all model files: weights, configs, tokenizer, and custom model code
# (custom code in Python files is needed for trust_remote_code=True at runtime)
snapshot_download(
    model_id,
    token=token,
    ignore_patterns=["*.ot", "*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
)

print(f"Downloaded {model_id} to /model_cache")

with open("/model_cache/MODEL_ID", "w") as f:
    f.write(model_id)

print("Done.")
