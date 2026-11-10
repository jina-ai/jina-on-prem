"""Register a checkpoint's own model classes with vLLM, in every process.

Some checkpoints ship a model class vLLM's built-in one cannot stand in for;
the catalog entry's ``vllm.architectures`` names each as ``"Name":
"module:Class"``, with ``module`` a top-level file of the snapshot. vLLM
resolves the model inside an engine process it spawns, which imports vLLM
from scratch, so a registration made only in the server would be invisible
there. This module is reached through the ``vllm.general_plugins`` entry
point instead, which vLLM calls in each process it starts; the metadata-only
package under ``docker/vllm-plugin`` declares it.

The snapshot directory goes on ``sys.path`` first, because the checkpoint's
files import each other by bare name and the class string resolves the same
way. Found through the cache rather than copied anywhere: the container runs
with ``HF_HUB_OFFLINE`` and the weights baked, so the lookup is local.
"""

import logging
import sys

logger = logging.getLogger(__name__)


def architectures(model_id: str) -> dict:
    """``{"Name": "module:Class"}`` for this model, or empty."""
    import catalog

    if not catalog.is_known(model_id):
        return {}
    return dict((catalog.spec_for(model_id).vllm or {}).get("architectures") or {})


def snapshot_dir(hf_repo: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(hf_repo, local_files_only=True)


def register() -> None:
    from catalog import spec_for
    from config import settings
    from vllm import ModelRegistry

    names = architectures(settings.model_id)
    if not names:
        return
    path = snapshot_dir(spec_for(settings.model_id).hf_repo)
    if path not in sys.path:
        sys.path.insert(0, path)
    for name, implementation in names.items():
        # By string, resolved lazily: nothing is imported until the engine
        # actually builds this architecture.
        ModelRegistry.register_model(name, implementation)
        logger.info(f"Registered {name} -> {implementation} with vLLM")
