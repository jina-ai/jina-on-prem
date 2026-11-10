"""Generative families: everything served at /v1/chat/completions."""

import logging
import os
from abc import abstractmethod

import torch

from config import settings

from . import Family

logger = logging.getLogger(__name__)


class ChatFamily(Family):
    """What the chat route needs to know about a model, and nothing about how it
    generates. Routing, the wrong-endpoint message and the text-only refusal read
    only these; each engine-specific subclass below owns the generation itself."""

    verb, kind, endpoint = "generate", "chat", "/v1/chat/completions"

    accepts_images = False


class TransformersChatFamily(ChatFamily):
    """Chat models driven through ``model.generate`` by ``engine.generate``."""

    # use_model_defaults was added in transformers >=4.51; reader-lm pins 4.48.3.
    generate_kwargs: dict = {}

    @abstractmethod
    def encode_prompt(self, conversation: list, images: list, max_length: int):
        """Template and tokenize the conversation into model inputs."""

    @property
    @abstractmethod
    def decoder_tokenizer(self):
        """Tokenizer used to decode generated ids back to text."""

    def _load_causal_lm(self) -> None:
        from transformers import AutoModelForCausalLM

        # Target dtype: fp16 on cuda (no bf16 native on L4), fp32 on cpu/mps.
        if settings.device == "cuda":
            dtype = (
                torch.bfloat16
                if settings.dtype in ("bfloat16", "bf16")
                else torch.float16
            )
        else:
            dtype = torch.float32
        # Flash-attn is intentionally not installed (no nvcc/git in the runtime
        # image); sdpa is the supported fallback and matches HF's documented
        # CPU/non-fa path.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.spec.hf_repo,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=settings.device if settings.device != "mps" else None,
            attn_implementation="sdpa",
        )
        if settings.device == "mps":
            self.model = self.model.to(settings.device)
        self.model.eval()
        logger.info(
            f"Loaded {self.spec.hf_repo} as {type(self).__name__} dtype={dtype}"
        )


class VlmFamily(TransformersChatFamily):
    accepts_images = True
    generate_kwargs = {"use_model_defaults": True}

    def _load(self) -> None:
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.spec.hf_repo, trust_remote_code=True, use_fast=False
        )
        self._load_causal_lm()

    def encode_prompt(self, conversation: list, images: list, max_length: int):
        # apply_chat_template returns the templated string for ONE conversation
        # when given a single list.
        texts = self.processor.apply_chat_template(
            [conversation], add_generation_prompt=True
        )
        return self.processor(
            text=texts,
            images=[images] if images else None,
            padding="longest",
            max_length=max_length,
            return_tensors="pt",
        )

    @property
    def decoder_tokenizer(self):
        return self.processor.tokenizer


class TextChatFamily(TransformersChatFamily):
    """Text-only chat / reader models: no vision tower, no AutoProcessor."""

    def _load(self) -> None:
        self._load_causal_lm()

    def encode_prompt(self, conversation: list, images: list, max_length: int):
        # AutoTokenizer's apply_chat_template only needs role + text.
        messages = [
            {
                "role": message["role"],
                "content": "".join(
                    part.get("text", "")
                    for part in message["content"]
                    if part.get("type") == "text"
                ),
            }
            for message in conversation
        ]
        templated = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return self.tokenizer(
            templated, return_tensors="pt", truncation=True, max_length=max_length
        )

    @property
    def decoder_tokenizer(self):
        return self.tokenizer


class VllmChatFamily(ChatFamily):
    """Chat models served by an in-process vLLM ``AsyncLLM``, through vLLM's
    own chat completions handler.

    Generic over models on purpose. Everything that differs between one model
    and the next is a value in the catalog entry's ``vllm`` block, spelled the
    way the hosted API's config spells it: ``architectures`` to register,
    ``engine`` (the ``AsyncEngineArgs`` fields, sampling defaults and the
    output ceiling included, under ``override_generation_config``) and
    ``serving`` (the loop stopper and the response options). This class owns
    the engine's lifecycle; ``vllm_serving`` builds vLLM's serving graph on
    top of it and the app mounts vLLM's route. Nothing here renders a prompt
    or reads a token back.

    ``accepts_images`` follows the catalog's ``modality``: the same family
    serves a page-reading model and a text-only one. A model that takes
    images declares how many under ``engine.limit_mm_per_prompt``, which is
    also the ceiling the route's image guard applies before the engine sees
    the request.
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.accepts_images = spec.modality == "multimodal"
        self.vllm = dict(spec.vllm or {})
        self.engine = None
        self.chat = None

    # --- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        # Not ``Family.load``: the hooks it runs after ``_load`` probe a
        # Transformers model this family never holds, and the chat template is
        # applied by vLLM's renderer, not here.
        self._load()

    def _load(self) -> None:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        # vLLM reads this from the environment only. Spawn, not fork: a forked
        # engine process would inherit this process's event loop and threads.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        config = AsyncEngineArgs(**self.engine_kwargs()).create_engine_config()
        self.engine = AsyncLLM.from_vllm_config(vllm_config=config, disable_log_stats=True)
        logger.info(f"Loaded {self.spec.hf_repo} as {type(self).__name__} on vLLM")

    def engine_kwargs(self) -> dict:
        """``AsyncEngineArgs`` fields: the process-level ones here, the
        model-level ones from the catalog, which win on any overlap. The one
        exception is an operator's ``JINA_GPU_MEMORY_FRACTION``, since that is
        how a second container shares the card."""
        kwargs = {
            "model": self.spec.hf_repo,
            "served_model_name": self.spec.id,
            "trust_remote_code": True,
            "dtype": "bfloat16" if settings.dtype in ("bfloat16", "bf16") else "float16",
        }
        kwargs.update(self.vllm.get("engine", {}))
        if settings.gpu_memory_fraction is not None:
            kwargs["gpu_memory_utilization"] = settings.gpu_memory_fraction
        return kwargs

    async def warmup(self) -> None:
        """Build vLLM's serving graph and prove the template carries a
        message, before the container reports ready."""
        import vllm_serving

        serving = vllm_serving.serving_config(self.vllm.get("serving"))
        chat = await vllm_serving.build_chat_serving(self.engine, self.spec.id, serving)
        await vllm_serving.warmup_chat_serving(chat, self.spec.id)
        self.chat = chat

    def alive(self) -> bool:
        # vLLM's monitor thread flips ``errored`` when the engine core dies;
        # nothing else in this process would notice.
        return self.engine is not None and not getattr(self.engine, "errored", False)

    def shutdown(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
