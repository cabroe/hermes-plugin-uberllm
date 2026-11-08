"""UberLLM image generation backend.

UberLLM (https://api.uberllm.dev/v1) serves image models on the OpenAI-compatible
``POST /v1/images/generations`` and ``POST /v1/images/edits`` routes, but its default
``response_format`` is ``url`` and the returned ``/v1/files/…/content`` link needs the
API key — which the bundled openai plugin does not send, so the image never lands in
Hermes' cache. This backend asks for ``b64_json`` instead and saves the bytes directly.

The catalog is read live from ``GET /v1/models?type=image`` (85 models at the time of
writing), so retired models drop out without a patch. Model ids are per capability:
text-only ids live on ``/v1/images/generations``, edit ids (``…-edit``) on
``/v1/images/edits`` and accept ``image_url`` / ``input_images``.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)
from agent.secret_scope import get_secret_str

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.uberllm.dev/v1"
DEFAULT_KEY_ENV = "HERMES_CUSTOM_API_UBERLLM_DEV_API_KEY"
DEFAULT_MODEL = "venice/z-image-turbo"
_REQUEST_TIMEOUT_S = 300.0
_CATALOG_TTL_S = 600.0
_MAX_REFERENCE_IMAGES = 4

# Preferred pixel sizes per semantic aspect, most-preferred first. UberLLM models
# advertise a ``size`` enum; the first entry the model actually lists wins.
_SIZE_PREFERENCE = {
    "landscape": ("1536x1024", "1024x768", "1344x768", "1024x1024"),
    "square": ("1024x1024", "768x768", "512x512"),
    "portrait": ("1024x1536", "768x1024", "1024x1024"),
}


def _cfg() -> dict[str, Any]:
    """``image_gen.uberllm`` from config.yaml; ``{}`` when absent or unreadable."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception as exc:  # config is optional; defaults are usable
        logger.debug("UberLLM image: cannot load config: %s", exc)
        return {}
    section = cfg.get("image_gen") if isinstance(cfg, dict) else None
    section = section if isinstance(section, dict) else {}
    own = section.get("uberllm")
    return own if isinstance(own, dict) else {}


def _base_url() -> str:
    return (
        str(_cfg().get("base_url") or os.environ.get("UBERLLM_IMAGE_BASE_URL") or DEFAULT_BASE_URL)
        .strip()
        .rstrip("/")
    )


def _api_key() -> str:
    key_env = str(_cfg().get("key_env") or DEFAULT_KEY_ENV).strip() or DEFAULT_KEY_ENV
    return (get_secret_str(key_env) or "").strip()


def _configured_model() -> str:
    own = str(_cfg().get("model") or "").strip()
    if own:
        return own
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            return str(section.get("model") or "").strip()
    except Exception:
        pass
    return ""


def _error_text(response: Any) -> str:
    try:
        payload = response.json()
    except Exception:
        return (getattr(response, "text", "") or "").strip()[:400]
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)[:400]
        if err:
            return str(err)[:400]
        if payload.get("message"):
            return str(payload["message"])[:400]
    return str(payload)[:400]


def _data_uri(value: str) -> str:
    """Inline a local file as a data URI; pass http(s)/data URIs through unchanged."""
    ref = str(value or "").strip()
    lower = ref.lower()
    if lower.startswith(("http://", "https://", "data:")):
        return ref
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    with open(ref, "rb") as fh:
        raw = fh.read()
    ext = os.path.splitext(ref)[1].lower()
    mime = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(ext, "image/jpeg")
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


class UberLLMImageGenProvider(ImageGenProvider):
    """Text-to-image and image editing through UberLLM's OpenAI-compatible image routes."""

    name = "uberllm"
    display_name = "UberLLM"

    def __init__(self) -> None:
        self._catalog_cache: Optional[tuple[list[dict[str, Any]], float]] = None
        self._detail_cache: dict[str, tuple[dict[str, Any], float]] = {}

    # ---- credentials / transport -------------------------------------------------------------
    def is_available(self) -> bool:
        return bool(_api_key())

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}

    def _session(self) -> Any:
        import requests

        return requests.Session()

    # ---- catalog -------------------------------------------------------------------------------
    def _catalog(self) -> list[dict[str, Any]]:
        if self._catalog_cache and time.monotonic() - self._catalog_cache[1] < _CATALOG_TTL_S:
            return self._catalog_cache[0]
        entries: list[dict[str, Any]] = []
        try:
            import requests

            response = requests.get(
                f"{_base_url()}/models",
                params={"type": "image"},
                headers={"Authorization": f"Bearer {_api_key()}"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json().get("data")
            entries = [e for e in (data if isinstance(data, list) else []) if isinstance(e, dict) and e.get("id")]
        except Exception as exc:
            logger.debug("UberLLM image catalog unavailable: %s", exc)
        self._catalog_cache = (entries, time.monotonic())
        return entries

    def _entry(self, model_id: str) -> dict[str, Any]:
        for entry in self._catalog():
            if str(entry.get("id")) == model_id:
                return entry
        return {}

    def _detail(self, model_id: str) -> dict[str, Any]:
        """Per-model detail (``parameters`` incl. the ``size`` enum); the list route omits it."""
        cached = self._detail_cache.get(model_id)
        if cached and time.monotonic() - cached[1] < _CATALOG_TTL_S:
            return cached[0]
        detail: dict[str, Any] = {}
        try:
            import requests

            response = requests.get(
                f"{_base_url()}/models/{model_id}",
                headers={"Authorization": f"Bearer {_api_key()}"},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            detail = payload if isinstance(payload, dict) else {}
        except Exception as exc:
            logger.debug("UberLLM image model detail unavailable for %s: %s", model_id, exc)
        self._detail_cache[model_id] = (detail, time.monotonic())
        return detail

    @staticmethod
    def _can_edit(entry: dict[str, Any]) -> bool:
        endpoints = entry.get("endpoints") or []
        return "/v1/images/edits" in endpoints or "image" in (entry.get("input_modalities") or [])

    def _available_ids(self) -> set[str]:
        return {str(e["id"]) for e in self._catalog() if e.get("available") is not False}

    def list_models(self) -> List[Dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for entry in self._catalog():
            if entry.get("available") is False:
                continue
            model_id = str(entry["id"])
            can_edit = self._can_edit(entry)
            pricing = entry.get("pricing") or {}
            price = ""
            if pricing.get("usd") is not None:
                price = f"${pricing['usd']}/{pricing.get('unit') or 'image'}"
            rows.append(
                {
                    "id": model_id,
                    "display": str(entry.get("display_name") or model_id.split("/")[-1]),
                    "strengths": "text-to-image + editing" if can_edit else "text-to-image",
                    "price": price,
                }
            )
        return sorted(rows, key=lambda m: m["id"])

    def default_model(self) -> Optional[str]:
        configured = _configured_model()
        if configured:
            return configured
        available = self._available_ids()
        if DEFAULT_MODEL in available:
            return DEFAULT_MODEL
        rows = self.list_models()
        return rows[0]["id"] if rows else None

    def capabilities(self) -> dict[str, Any]:
        """Advertise editing only when the configured model can actually take a source image.

        The tool schema is built from this, so claiming ``image`` for a text-only model
        would invite calls that can only fail.
        """
        configured = _configured_model() or self.default_model() or ""
        can_edit = self._can_edit(self._entry(configured)) if configured else False
        return {
            "modalities": ["text", "image"] if can_edit else ["text"],
            "max_reference_images": _MAX_REFERENCE_IMAGES if can_edit else 0,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "UberLLM",
            "badge": "paid",
            "tag": "Flux, Seedream, Nano Banana, Ideogram, Krea … — live catalog from api.uberllm.dev; "
                   "text-to-image and editing, images saved as b64_json",
            "env_vars": [
                {
                    "key": DEFAULT_KEY_ENV,
                    "prompt": "UberLLM API key",
                    "url": "https://uberllm.dev/dashboard/keys",
                }
            ],
        }

    # ---- request shaping -----------------------------------------------------------------------
    def _size_for(self, model_id: str, aspect: str) -> Optional[str]:
        props = ((self._detail(model_id).get("parameters") or {}).get("properties") or {})
        allowed = (props.get("size") or {}).get("enum")
        if not isinstance(allowed, list) or not allowed:
            return None
        for candidate in _SIZE_PREFERENCE.get(aspect, _SIZE_PREFERENCE["square"]):
            if candidate in allowed:
                return candidate
        return str(allowed[0])

    def _resolve_model(self, *, editing: bool) -> str:
        configured = _configured_model() or self.default_model() or ""
        available = self._available_ids()
        if not configured:
            raise ValueError(
                "No UberLLM image model configured. Set one with "
                "`hermes config set image_gen.uberllm.model <id>` (see `hermes tools` → Image Generation)."
            )
        entry = self._entry(configured)
        if editing and not self._can_edit(entry):
            edits = sorted(
                (
                    ((e.get("pricing") or {}).get("usd"), str(e["id"]))
                    for e in self._catalog()
                    if e.get("available") is not False and self._can_edit(e)
                ),
                key=lambda pair: (pair[0] is None, pair[0] if pair[0] is not None else 0.0),
            )
            if edits:
                cheapest = edits[0][1]
                raise ValueError(
                    f"UberLLM model {configured!r} is text-to-image only and cannot take a source image. "
                    f"Pick an edit model, e.g. `hermes config set image_gen.uberllm.model {cheapest}`"
                )
            raise ValueError(
                f"UberLLM model {configured!r} is text-to-image only and no edit model is available."
            )
        if available and configured not in available:
            raise ValueError(
                f"UberLLM image model {configured!r} is not available right now. "
                "Pick another with `hermes config set image_gen.uberllm.model <id>`."
            )
        return configured

    # ---- generation ----------------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return error_response(
                error="A prompt is required for image generation.",
                error_type="invalid_request",
                provider=self.name,
                aspect_ratio=aspect,
            )
        if not _api_key():
            return error_response(
                error=f"{DEFAULT_KEY_ENV} is not set (or the variable named by "
                      "image_gen.uberllm.key_env is empty).",
                error_type="auth_required",
                provider=self.name,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        sources: list[str] = []
        if isinstance(image_url, str) and image_url.strip():
            sources.append(image_url.strip())
        for ref in reference_image_urls or []:
            if isinstance(ref, str) and ref.strip():
                sources.append(ref.strip())
        sources = sources[:_MAX_REFERENCE_IMAGES]
        editing = bool(sources)

        try:
            model_id = self._resolve_model(editing=editing)
        except ValueError as exc:
            return error_response(
                error=str(exc), error_type="config_error", provider=self.name,
                prompt=prompt, aspect_ratio=aspect,
            )

        body: dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
        }
        size = self._size_for(model_id, aspect)
        if size:
            body["size"] = size
        if editing:
            try:
                inlined = [_data_uri(ref) for ref in sources]
            except Exception as exc:
                return error_response(
                    error=f"Could not read source image: {exc}", error_type="io_error",
                    provider=self.name, model=model_id, prompt=prompt, aspect_ratio=aspect,
                )
            if len(inlined) == 1:
                body["image_url"] = inlined[0]
            else:
                body["input_images"] = inlined

        route = "/images/edits" if editing else "/images/generations"
        session = self._session()
        try:
            response = session.post(
                f"{_base_url()}{route}", json=body, headers=self._headers(), timeout=_REQUEST_TIMEOUT_S
            )
        except Exception as exc:
            logger.debug("UberLLM image request failed", exc_info=True)
            return error_response(
                error=f"UberLLM image request failed: {exc}", error_type="network_error",
                provider=self.name, model=model_id, prompt=prompt, aspect_ratio=aspect,
            )
        if response.status_code >= 400:
            detail = _error_text(response)
            kind = "auth_required" if response.status_code in (401, 403) else "api_error"
            return error_response(
                error=f"UberLLM image {'editing' if editing else 'generation'} failed "
                      f"(HTTP {response.status_code}): {detail}",
                error_type=kind, provider=self.name, model=model_id,
                prompt=prompt, aspect_ratio=aspect,
            )
        try:
            payload = response.json()
        except Exception:
            return error_response(
                error="UberLLM returned a non-JSON image response.", error_type="api_error",
                provider=self.name, model=model_id, prompt=prompt, aspect_ratio=aspect,
            )

        items = payload.get("data") if isinstance(payload, dict) else None
        first = items[0] if isinstance(items, list) and items else {}
        if not isinstance(first, dict):
            first = {}
        b64 = first.get("b64_json")
        url = first.get("url")
        image_ref: Optional[str] = None
        if b64:
            try:
                image_ref = str(save_b64_image(b64, prefix=f"uberllm_{model_id.split('/')[-1]}"))
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}", error_type="io_error",
                    provider=self.name, model=model_id, prompt=prompt, aspect_ratio=aspect,
                )
        elif url:
            # Fallback: the file URL needs the API key, so fetch it ourselves.
            try:
                import requests

                fetched = requests.get(url, headers={"Authorization": f"Bearer {_api_key()}"}, timeout=120)
                fetched.raise_for_status()
                image_ref = str(
                    save_b64_image(
                        base64.b64encode(fetched.content).decode(),
                        prefix=f"uberllm_{model_id.split('/')[-1]}",
                    )
                )
            except Exception as exc:
                logger.debug("UberLLM image URL download failed: %s", exc)
                return error_response(
                    error=f"UberLLM returned an image URL that could not be downloaded: {exc}",
                    error_type="io_error", provider=self.name, model=model_id,
                    prompt=prompt, aspect_ratio=aspect,
                )
        else:
            return error_response(
                error="UberLLM response contained neither b64_json nor a URL.",
                error_type="empty_response", provider=self.name, model=model_id,
                prompt=prompt, aspect_ratio=aspect,
            )

        usage = payload.get("usage") if isinstance(payload, dict) else None
        extra: dict[str, Any] = {"size": size or ""}
        if isinstance(usage, dict) and usage.get("cost_usd") is not None:
            extra["cost_usd"] = usage["cost_usd"]
        if first.get("revised_prompt"):
            extra["revised_prompt"] = first["revised_prompt"]
        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            modality="image" if editing else "text",
            extra=extra,
        )
