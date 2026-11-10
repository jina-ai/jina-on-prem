"""vLLM's own chat completions, mounted into this app.

vLLM ships chat completions as a mountable ``APIRouter`` over a plain object
graph, and the graph can be built from ordinary values rather than from the
``argparse.Namespace`` its own server builds. That is what this module does:
it builds ``OpenAIServingChat`` against the ``AsyncLLM`` the family already
owns and mounts vLLM's router into this app, the same way the hosted API
serves the same model.

Nothing here reimplements chat: templating, SSE framing, ``finish_reason``
semantics, ``usage``, and client-disconnect cancellation are vLLM's, and are
better tested than anything written here would be. This module is the wiring,
plus the parts vLLM's own entrypoint does that have to be done differently
here: the output ceiling is refused rather than clamped, images are resolved
by this server's media layer rather than fetched inside the engine, and the
engine's exceptions are answered the way vLLM's server answers them.

vLLM is imported lazily inside the functions that touch it, so this module
loads without it; the test suite fakes it at that boundary.
"""

import asyncio
import base64
import logging
from http import HTTPStatus
from typing import Annotated, Any, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

import media
from errors import BadRequest, JinaError

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# Catalog ``vllm.serving`` keys and their defaults, spelled as the hosted
# API's serving config spells them. ``force_include_usage`` puts a running
# ``usage`` on every streamed chunk, so an aborted stream is still accounted
# for from the last frame the client saw.
SERVING_DEFAULTS = {
    "repetition_detection": None,
    "chat_template": None,
    "chat_template_content_format": "auto",
    "chat_template_kwargs": {},
    "response_role": "assistant",
    "enable_prompt_tokens_details": False,
    "force_include_usage": False,
}


def image_limit(spec: Any) -> Optional[int]:
    """How many images a request may carry: the engine's own
    ``limit_mm_per_prompt`` for a multimodal model, None for a text-only one.

    Undeclared on a multimodal model there is no honest number to fall back
    to: zero refuses every image on a model configured to serve them, and
    anything else invents an admission policy out of a missing key.
    """
    if spec.modality != "multimodal":
        return None
    limits = (spec.vllm or {}).get("engine", {}).get("limit_mm_per_prompt") or {}
    if "image" not in limits:
        raise ValueError(
            f"{spec.id}: a multimodal model on vLLM needs vllm.engine.limit_mm_per_prompt.image, "
            f"the per-request ceiling the image guard enforces before the engine"
        )
    return limits["image"]


def serving_config(block: Optional[dict]) -> dict:
    """The catalog's ``serving`` block over the defaults, refused by name at
    start-up for anything the hosted API's config would refuse too."""
    block = dict(block or {})
    unknown = sorted(set(block) - set(SERVING_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown vllm.serving keys: {unknown}")
    detection = block.get("repetition_detection")
    if detection is not None:
        keys = {"max_pattern_size", "min_pattern_size", "min_count"}
        if set(detection) != keys:
            raise ValueError(
                f"repetition_detection needs exactly {sorted(keys)}, got {sorted(detection)}"
            )
        if (
            min(detection["max_pattern_size"], detection["min_pattern_size"]) < 1
            or detection["min_count"] < 2
        ):
            raise ValueError("repetition_detection: pattern sizes must be >= 1 and min_count >= 2")
        if detection["min_pattern_size"] > detection["max_pattern_size"]:
            raise ValueError("repetition_detection: min_pattern_size must be <= max_pattern_size")
    return {**SERVING_DEFAULTS, **block}


def repetition_params(block: dict) -> Any:
    """vLLM's n-gram loop stop, applied as a server-side default.

    A fresh ``RepetitionDetectionParams`` per request rather than one shared
    object, because vLLM's is a plain mutable dataclass that every in-flight
    ``SamplingParams`` would otherwise share. It is a stop condition, not a
    sampler: it sits beside the EOS and ``max_tokens`` checks and never
    touches logits, so it cannot change what a page that does not loop
    transcribes to. It cannot live in ``override_generation_config`` with the
    other sampling defaults, because vLLM filters that mapping down to a fixed
    set of names and drops the rest, so the default is written onto the
    request instead.
    """
    from vllm.sampling_params import RepetitionDetectionParams

    return RepetitionDetectionParams(
        max_pattern_size=block["max_pattern_size"],
        min_pattern_size=block["min_pattern_size"],
        min_count=block["min_count"],
    )


async def build_chat_serving(engine: Any, served_model_name: str, serving: dict) -> Any:
    """Construct vLLM's chat serving graph against an already-built ``AsyncLLM``.

    Mirrors what vLLM's own ``build_app`` does, minus everything tied to its
    CLI: ``OpenAIServingModels`` (name resolution) feeds ``OpenAIServingRender``
    (prompt rendering) feeds ``OpenAIServingChat`` (the request handler).
    ``request_logger`` is None throughout; this server's access log is the one
    source of request logs.
    """
    from vllm.entrypoints.chat_utils import load_chat_template
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender

    # The published name, not the repo id: ``OpenAIServingModels`` compares the
    # request's ``model`` by exact string and answers 404 on a miss, so this
    # is the one string a client is allowed to send.
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=[BaseModelPath(name=served_model_name, model_path=served_model_name)],
        lora_modules=None,
    )
    await models.init_static_loras()

    # The serving objects take the template's TEXT, not a path to it. Passing
    # the path through is silently wrong rather than an error: the path is
    # itself a valid Jinja template that renders to itself.
    chat_template = load_chat_template(serving["chat_template"])
    template_kwargs = dict(serving["chat_template_kwargs"])
    render = OpenAIServingRender(
        model_config=engine.model_config,
        renderer=engine.renderer,
        model_registry=models.registry,
        request_logger=None,
        chat_template=chat_template,
        chat_template_content_format=serving["chat_template_content_format"],
        default_chat_template_kwargs=template_kwargs,
    )
    kwargs = dict(
        engine_client=engine,
        models=models,
        response_role=serving["response_role"],
        openai_serving_render=render,
        request_logger=None,
        chat_template=chat_template,
        chat_template_content_format=serving["chat_template_content_format"],
        default_chat_template_kwargs=template_kwargs,
        enable_prompt_tokens_details=serving["enable_prompt_tokens_details"],
        enable_force_include_usage=serving["force_include_usage"],
    )

    detection = serving["repetition_detection"]

    class _ChatWithOutputLimits(OpenAIServingChat):
        """``OpenAIServingChat`` with the two output limits vLLM cannot express
        itself: the loop stopper as a default, and the ceiling as a refusal."""

        async def create_chat_completion(self, request, raw_request=None):
            refusal = self._refuse_oversized_output(request)
            if refusal is not None:
                return refusal
            if detection is not None and request.repetition_detection is None:
                request.repetition_detection = repetition_params(detection)
            return await super().create_chat_completion(request, raw_request)

        def _refuse_oversized_output(self, request):
            """Refuse an output cap above what this model serves, instead of
            clamping.

            vLLM mins the request against ``override_generation_config``'s
            ``max_new_tokens``, so a caller asking for more is silently held
            to the ceiling and told the answer finished on its own, which is
            the reading of a truncated page that costs the most to discover.
            vLLM refuses the same mistake at the context window and names the
            field; this gives the nearer limit the same manners.
            """
            ceiling = self.override_max_tokens
            explicit = request.max_completion_tokens is not None
            requested = request.max_completion_tokens if explicit else request.max_tokens
            if ceiling is None or requested is None or requested <= ceiling:
                return None
            field = "max_completion_tokens" if explicit else "max_tokens"
            return self.create_error_response(
                f"{field}={requested} cannot be greater than {ceiling}, the most this "
                "model will generate. Please request fewer output tokens, or omit the "
                "field to accept the model's own limit.",
                param=field,
            )

    return _ChatWithOutputLimits(**kwargs)


def mount_chat_completions(app: Any, *, max_images: Optional[int]) -> None:
    """Mount vLLM's chat router onto this app.

    The router resolves its handler from ``app.state.openai_serving_chat`` on
    every request, not at mount time, so the route is declared here at
    construction and the handler is filled in from the app's startup once the
    engine is warm. Registered before this server's own chat route, so it is
    the one that matches for a model served on vLLM.

    vLLM's router also carries ``/v1/chat/completions/batch``, whose handler
    reads an ``app.state`` attribute never set here. Only the chat route is
    taken, by selecting it out of vLLM's router *before* mounting: from
    FastAPI 0.140 ``include_router`` no longer flattens the included routes,
    so a post-hoc filter matches nothing and quietly mounts everything.

    ``max_images`` is the engine's ``limit_mm_per_prompt`` for images; with
    it, images are resolved through this server's media layer before vLLM
    sees the request. None means a text-only model, and no guard.
    """
    from fastapi import APIRouter
    from vllm.entrypoints.openai.chat_completion.api_router import router as vllm_router

    served = APIRouter()
    served.routes = [
        route
        for route in vllm_router.routes
        if getattr(route, "path", None) == CHAT_COMPLETIONS_PATH
    ]
    if not served.routes:
        raise RuntimeError(
            f"vLLM's chat router has no {CHAT_COMPLETIONS_PATH} route "
            f"(found {[getattr(r, 'path', r) for r in vllm_router.routes]})"
        )

    app.state.openai_serving_chat = None
    if max_images is not None:
        # Registered before vLLM's router is included, so it matches first. It
        # rewrites the images and then calls vLLM's own handler, not a copy of
        # it, under vLLM's own route dependencies, so everything from the
        # rewrite onward is vLLM's code on vLLM's contract.
        _mount_guarded_image_route(app, served.routes[0], max_images)
    app.include_router(served)


def _mount_guarded_image_route(app: Any, route: Any, max_images: int) -> None:
    """Shadow the chat path with one that resolves images before vLLM sees them.

    Takes vLLM's route rather than just its handler so the shadow carries the
    same route-level dependencies: ``validate_json_request`` is what makes
    this endpoint refuse a body sent as anything but ``application/json``.
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.serve.utils.error_response import create_error_response

    async def chat_completions_with_guarded_images(request, raw_request):
        try:
            await inline_images(request.messages, limit=max_images)
        except JinaError as exc:
            # The media layer speaks in this server's envelope. Restated through
            # vLLM's own helper so a rejected image reads like every other 4xx
            # this endpoint produces, rather than like a different API.
            error = create_error_response(
                exc.message, status_code=HTTPStatus(exc.status), param=exc.field
            )
            return JSONResponse(error.model_dump(), status_code=exc.status)
        return await route.endpoint(request, raw_request)

    # Assigned rather than written in the signature: both names are local
    # imports, and a string annotation FastAPI cannot resolve is treated as a
    # body field rather than reported, so every request would answer 422.
    chat_completions_with_guarded_images.__annotations__ = {
        "request": ChatCompletionRequest,
        "raw_request": Request,
        "return": Any,
    }
    app.post(
        CHAT_COMPLETIONS_PATH,
        include_in_schema=False,
        dependencies=route.dependencies,
    )(chat_completions_with_guarded_images)


def register_vllm_exception_handlers(app: Any) -> None:
    """Answer vLLM's exceptions the way vLLM's own server would.

    Mounting vLLM's router here means the handlers its ``build_app`` installs
    are never registered, so everything the chat route raises would escape to
    Starlette uncaught: a rejected second image, a blown chat template and a
    dead engine would all reach the client as the same bodyless 500.

    The exception-to-status mapping is deliberately not ours:
    ``create_error_response`` is vLLM's and a pure function of the exception.
    Its handlers are not reusable, only that function: each one reads an
    ``app.state.args`` that only vLLM's own entrypoint sets.

    Naming the types as well as ``Exception`` is not redundant: Starlette
    sends an ``Exception`` handler to the outermost middleware and every other
    type to the innermost, and only the inner one answers *below* the
    middleware, which is what lets a 400 be recorded as a 400. The backstop
    does not repeat what it caught: for an unrecognised exception the message
    is a snapshot path, a CUDA allocator message or an assertion from inside
    the engine, and no other route here answers an unhandled error with more
    than a fixed string.
    """
    from fastapi.responses import JSONResponse
    from jinja2 import TemplateError
    from vllm.entrypoints.openai.engine.protocol import GenerationError
    from vllm.entrypoints.serve.utils.error_response import create_error_response
    from vllm.exceptions import VLLMValidationError
    from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

    def log(request, exc, status_code):
        message = f"{type(exc).__name__} on {request.url.path}: {exc}"
        # A 4xx is the caller's; a stack trace per malformed request is noise.
        if status_code >= 500:
            logger.exception(message)
        else:
            logger.warning(message)

    async def handle(request, exc):
        error = create_error_response(exc)
        log(request, exc, error.error.code)
        return JSONResponse(error.model_dump(), status_code=error.error.code)

    async def handle_unrecognised(request, exc):
        error = create_error_response(exc)
        log(request, exc, error.error.code)
        error.error.message = "Internal server error"
        return JSONResponse(error.model_dump(), status_code=error.error.code)

    for exc_type in (
        # vLLM's 400s. The message is a description of the caller's own request.
        ValueError,
        TypeError,
        OverflowError,
        TemplateError,
        VLLMValidationError,
        # vLLM's 500s, but named ones: the text is a fixed string of vLLM's.
        EngineGenerateError,
        EngineDeadError,
        GenerationError,
    ):
        app.exception_handler(exc_type)(handle)
    app.exception_handler(Exception)(handle_unrecognised)


# A string no chat template can produce by itself, so finding it in the
# rendered prompt proves the message survived templating rather than that
# something plausible came out the other end.
WARMUP_SENTINEL = "jina-warmup-probe-0e6a"


async def warmup_chat_serving(chat: Any, model_name: str) -> None:
    """Warm the renderer, then prove the template actually carries the message.

    ``chat.warmup()`` is vLLM's own renderer warm-up. The generation after it
    is ours, and the last assertion is the whole reason it exists: a template
    variable only the checkpoint's processor supplies renders as Jinja
    undefined, which becomes an empty string rather than an error, and the
    model then answers a prompt missing its content, fluently and wrongly.
    Checking ``error`` and ``choices`` cannot catch that, so the prompt is read
    back and asserted to contain what was sent. Costs one prefill of a few
    tokens plus a single decode step.
    """
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    chat.warmup()

    request = ChatCompletionRequest(
        model=model_name,
        messages=[{"role": "user", "content": WARMUP_SENTINEL}],
        max_completion_tokens=1,
        temperature=0.0,
        stream=False,
        return_prompt_text=True,
    )
    result = await chat.create_chat_completion(request)

    error = getattr(result, "error", None)
    if error is not None:
        raise RuntimeError(f"chat warmup returned an error response: {error}")
    if not getattr(result, "choices", None):
        raise RuntimeError(f"chat warmup produced no choices: {result!r}")
    prompt = getattr(result, "prompt_text", None)
    if prompt is None:
        # Refusing to boot is the point: a warning here leaves the template
        # unchecked with nothing to show that it was not checked.
        raise RuntimeError(
            "chat warmup could not read the rendered prompt back, so the template "
            "check did not run; `return_prompt_text` is set and vLLM should have "
            "populated `prompt_text`"
        )
    if WARMUP_SENTINEL not in prompt:
        raise RuntimeError(
            f"the chat template dropped the message content; rendered {prompt!r} "
            f"without {WARMUP_SENTINEL!r}"
        )


_IGNORE = ConfigDict(extra="ignore")


class TextContentPart(BaseModel):
    model_config = _IGNORE

    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    model_config = _IGNORE

    url: str


class ImageContentPart(BaseModel):
    model_config = _IGNORE

    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Annotated[Union[TextContentPart, ImageContentPart], Field(discriminator="type")]

_PARTS = TypeAdapter(List[ContentPart])

Position = Tuple[int, int]


def image_parts(messages: Any, *, limit: int) -> List[Tuple[Position, str]]:
    """Locate every image, refusing content this model will not serve.

    ``limit`` is the engine's own ``limit_mm_per_prompt``, restated here
    because the engine applies it *after* templating and the fetch happens
    first. Without it, a small body naming hundreds of URLs starts hundreds
    of downloads, each entitled to the media limit, before anything counts
    them. A message may carry a bare string instead of a list of parts, which
    has no part to inspect by construction.
    """
    found: List[Tuple[Position, str]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        try:
            parts = _PARTS.validate_python(content)
        except ValidationError as exc:
            raise BadRequest(_first_reason(exc, message_index))
        found.extend(
            ((message_index, part_index), part.image_url.url)
            for part_index, part in enumerate(parts)
            if isinstance(part, ImageContentPart)
        )

    if len(found) > limit:
        raise BadRequest(f"{len(found)} images in one request exceeds the limit of {limit}.")
    return found


def _first_reason(error: ValidationError, message_index: int) -> str:
    """The one thing wrong, located the way the client can find it in its own
    request body."""
    first = error.errors()[0]
    location = ".".join(str(step) for step in first["loc"])
    return f"messages.{message_index}.content.{location}: {first['msg']}"


async def inline_images(messages: Any, *, limit: int) -> int:
    """Replace every image reference with the bytes it resolves to.

    vLLM's chat serving fetches ``image_url`` parts itself, from inside the
    engine and through none of this server's code: no scheme check, no cap on
    the payload, no bound on what a header-sized PNG expands to. So they are
    resolved here through the same media layer every other endpoint uses and
    each part is rewritten into a ``data:`` URI. The bytes in that URI are the
    client's own: vLLM applies EXIF orientation and composites transparency
    onto white on decode, and re-encoding here would quietly change the
    pixels the model reads. Inline ``data:`` URIs are rewritten too, so the
    caps reach them as well, and so the media type is canonical by the time
    vLLM parses it.

    Content is validated against a schema (``image_parts``), not searched for
    images: vLLM's parser accepts a much wider union than is served here, and
    the surplus is not inert, since a part with no ``type`` reaches a media
    loader inside the engine.

    Mutates in place: the request object was parsed for this call and nobody
    else holds it. Returns how many parts were rewritten, which is 0 for a
    text-only request, the shape that must stay free. A bad, blocked,
    unfetchable or oversized reference raises what the media layer raises.
    """
    at = image_parts(messages, limit=limit)
    if not at:
        return 0

    fetches = [asyncio.ensure_future(asyncio.to_thread(media.load_bytes, url)) for _, url in at]
    try:
        resolved = await asyncio.gather(*fetches)
    except BaseException:
        # ``gather`` raises on the first failure and leaves the rest running.
        for fetch in fetches:
            fetch.cancel()
        raise

    for ((message, part), _url), (data, media_type) in zip(at, resolved):
        messages[message]["content"][part] = {
            "type": "image_url",
            "image_url": {"url": await asyncio.to_thread(_data_uri, data, media_type)},
        }
    return len(at)


def _data_uri(data: bytes, media_type: str) -> str:
    """Off the event loop: base64 of a media-limit image is tens of
    milliseconds, and this loop also carries every stream's tokens."""
    return f"data:{media_type};base64,{base64.b64encode(data).decode()}"
