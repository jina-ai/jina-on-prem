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

# Patch model code in all cached locations for air-gap compatibility.
# These patches fix bugs in the HF model code that break offline loading.
import glob

# 1. custom_st.py: Multiple patches for air-gap compatibility
import re
for custom_st in glob.glob("/model_cache/**/custom_st.py", recursive=True):
    with open(custom_st, "r") as f:
        src = f.read()
    original = src

    # 1a. Fix default_task extraction (v3 model_args.pop / v5 model_kwargs.pop)
    src = re.sub(
        r"^\s*self\.default_task\s*=\s*model_(?:args|kwargs)\.pop\([^)]*\).*$",
        "        self.default_task = None  # patched for air-gap",
        src,
        flags=re.MULTILINE,
    )

    # 1b. Add trust_remote_code=True to AutoConfig.from_pretrained
    src = src.replace(
        'self.config = AutoConfig.from_pretrained(\n            model_name_or_path, **config_kwargs\n        )',
        'config_kwargs["trust_remote_code"] = True\n        self.config = AutoConfig.from_pretrained(\n            model_name_or_path, **config_kwargs\n        )'
    )

    # 1c. Add trust_remote_code=True to AutoModel.from_pretrained
    src = src.replace(
        'self.model = AutoModel.from_pretrained(\n            model_name_or_path, config=self.config, **model_kwargs\n        )',
        'model_kwargs["trust_remote_code"] = True\n        self.model = AutoModel.from_pretrained(\n            model_name_or_path, config=self.config, **model_kwargs\n        )'
    )

    if src != original:
        with open(custom_st, "w") as f:
            f.write(src)
        print(f"Patched {custom_st}: fixed default_task + trust_remote_code")

# 2. modeling_eurobert.py: EuroBertModel.__init__ needs **kwargs
#    because modeling_jina_embeddings_v5.py passes dtype= to from_pretrained,
#    and transformers 4.48.x forwards unknown kwargs to __init__.
for eurobert in glob.glob("/model_cache/**/modeling_eurobert.py", recursive=True):
    with open(eurobert, "r") as f:
        src = f.read()
    original = src
    old_init = '    def __init__(self, config: EuroBertConfig):\n        super().__init__(config)'
    new_init = '    def __init__(self, config: EuroBertConfig, **kwargs):\n        super().__init__(config)'
    src = src.replace(old_init, new_init)
    if src != original:
        with open(eurobert, "w") as f:
            f.write(src)
        print(f"Patched {eurobert}: added **kwargs to EuroBertModel.__init__")

# 3. modeling_jina_embeddings_v5.py: multiple patches
for modeling in glob.glob("/model_cache/**/modeling_jina_embeddings_v5.py", recursive=True):
    with open(modeling, "r") as f:
        src = f.read()
    original = src
    # 3a. PeftConfig -> LoraConfig
    src = src.replace(
        'from peft import PeftMixedModel, PeftConfig',
        'from peft import PeftMixedModel, PeftConfig, LoraConfig'
    )
    src = src.replace(
        'peft_config = PeftConfig.from_pretrained(',
        'peft_config = LoraConfig.from_pretrained('
    )
    # 3b. Fix dtype -> torch_dtype for Qwen3Model.from_pretrained
    #     transformers >=4.51.0 Qwen3Model doesn't accept 'dtype' kwarg
    src = src.replace(
        'dtype=kwargs.pop("dtype", torch.bfloat16)',
        'torch_dtype=kwargs.pop("dtype", torch.bfloat16)'
    )
    if src != original:
        with open(modeling, "w") as f:
            f.write(src)
        print(f"Patched {modeling}: PeftConfig->LoraConfig + dtype->torch_dtype")

with open("/model_cache/MODEL_ID", "w") as f:
    f.write(model_id)

print("Done.")
