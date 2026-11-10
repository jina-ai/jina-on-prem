"""Bake a vLLM-served model's weights into the image, in the build's downloader step.

Only what the weight index references is pulled, plus the config, tokenizer
and code files. A checkpoint can carry a second, unreferenced shard set and
training artefacts alongside the release weights, and taking the whole repo
costs several times what gets loaded. The shard names come out of the index
rather than a pattern, so a release that reshards still builds; every
referenced file is then asserted present, because a download filtered down to
nothing exits 0 and fails much later as a missing-weights error.

Writes `refs/main` and `/model_cache/MODEL_ID` the way download_model.py does,
so the server resolves the pinned snapshot offline and finds the model id
without JINA_MODEL_ID. The transformers pre-load that script runs is not
wanted here: vLLM loads the checkpoint itself.
"""

import json
import os
import pathlib
import sys

os.environ["HF_HOME"] = "/model_cache"

token = os.environ.get("HF_TOKEN") or None
model_id = os.environ.get("MODEL_ID", "")
revision = os.environ.get("MODEL_REVISION", "").strip() or None
if not model_id:
    print("ERROR: MODEL_ID env var not set", file=sys.stderr)
    sys.exit(1)

from huggingface_hub import hf_hub_download, snapshot_download

INDEX = "model.safetensors.index.json"
METADATA = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "*.py",
]

print(f"Downloading {model_id}" + (f" @ {revision}" if revision else " @ main (UNPINNED)"))
try:
    index_path = hf_hub_download(model_id, INDEX, revision=revision, token=token)
    shards = sorted(set(json.loads(pathlib.Path(index_path).read_text())["weight_map"].values()))
    wanted = [INDEX, *shards]
except Exception:
    # A single-file checkpoint has no index.
    shards, wanted = ["model.safetensors"], ["model.safetensors"]

snapshot = pathlib.Path(
    snapshot_download(
        model_id, revision=revision, token=token, allow_patterns=[*METADATA, *wanted]
    )
)
missing = [name for name in wanted if not (snapshot / name).is_file()]
if missing:
    print(f"ERROR: weight download produced no {missing}", file=sys.stderr)
    sys.exit(1)

if revision:
    # snapshot_download writes refs/<branch> only for a branch name. The
    # server resolves "main" through refs/ offline, so pin it ourselves.
    refs = pathlib.Path("/model_cache/hub") / f"models--{model_id.replace('/', '--')}" / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision)

# transformers would pip-install from this at load time; the image is pinned.
for requirements in snapshot.rglob("requirements.txt"):
    requirements.unlink()

size = sum(path.stat().st_size for path in snapshot.rglob("*") if path.is_file())
print(f"Baked {len(shards)} shard(s), {size / 2**30:.2f} GiB, into {snapshot}")

with open("/model_cache/MODEL_ID", "w") as f:
    f.write(model_id)
