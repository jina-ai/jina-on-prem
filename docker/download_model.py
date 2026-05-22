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
snapshot_download(
    model_id,
    token=token,
    ignore_patterns=["*.ot", "*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
)

print(f"Downloaded {model_id} to /model_cache")

# Pre-load with SentenceTransformer to cache dynamic modules
# This ensures trust_remote_code modules are in the HF cache for offline use
try:
    from sentence_transformers import SentenceTransformer
    print(f"Pre-loading {model_id} to cache dynamic modules...")
    SentenceTransformer(model_id, trust_remote_code=True, device="cpu")
    print("Dynamic modules cached successfully")
except Exception as e:
    print(f"Warning: pre-load failed, model may still work at runtime: {e}")

# Patch custom_st.py in all cached locations:
# 1. model_args/config_args default None -> {} to avoid .pop() on None
# 2. Add trust_remote_code=True to AutoConfig/AutoModel calls
import glob
for custom_st in glob.glob("/model_cache/**/custom_st.py", recursive=True):
    with open(custom_st, "r") as f:
        src = f.read()
    original = src
    # Fix model_args.pop on None: change to use model_kwargs which has or {} fallback
    src = src.replace('self.default_task = model_args.pop(', 'self.default_task = model_kwargs.pop(')
    # Add trust_remote_code=True to AutoConfig/AutoModel calls
    src = src.replace(
        'self.config = AutoConfig.from_pretrained(\n            model_name_or_path, **config_kwargs\n        )',
        'config_kwargs["trust_remote_code"] = True\n        self.config = AutoConfig.from_pretrained(\n            model_name_or_path, **config_kwargs\n        )'
    )
    src = src.replace(
        'self.model = AutoModel.from_pretrained(\n            model_name_or_path, config=self.config, **model_kwargs\n        )',
        'model_kwargs["trust_remote_code"] = True\n        self.model = AutoModel.from_pretrained(\n            model_name_or_path, config=self.config, **model_kwargs\n        )'
    )
    if src != original:
        with open(custom_st, "w") as f:
            f.write(src)
        print(f"Patched {custom_st}")
    else:
        print(f"No patch needed: {custom_st}")

with open("/model_cache/MODEL_ID", "w") as f:
    f.write(model_id)

print("Done.")
