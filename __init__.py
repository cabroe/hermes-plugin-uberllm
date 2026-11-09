"""UberLLM video generation backend.

UberLLM (https://api.uberllm.dev/v1) serves video models on its own async-job route:
``POST /v1/video/generations`` → ``202 {id, poll_url}`` → ``GET /v1/video/generations/<id>``
until ``succeeded`` → download ``results[0].url`` with the API key.

Model ids are per modality (``…-text-to-video``, ``…-image-to-video``,
``…-reference-to-video``), so this backend maps the configured family to the id that
matches the call: ``image_url`` → image-to-video, ``reference_image_urls`` →
reference-to-video, otherwise text-to-video. The catalog is read live from
``GET /v1/models?type=video`` so retired models drop out without a patch.
"""

from __future__ import annotations

import base64
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from agent.secret_scope import get_secret_str
from agent.video_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    VideoGenProvider,
    error_response,
    save_url_video,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.uberllm.dev/v1"
DEFAULT_KEY_ENV = "HERMES_CUSTOM_API_UBERLLM_DEV_API_KEY"
_SUFFIXES = ("-text-to-video", "-image-to-video", "-reference-to-video")
_POLL_INTERVAL_S = 5.0
_POLL_DEADLINE_S = 900.0
_REQUEST_TIMEOUT_S = 60.0
_CATALOG_TTL_S = 600.0


def _cfg() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception as exc:  # config is optional; defaults are usable
        logger.debug("UberLLM video: cannot load config: %s", exc)
        return {}
    section = cfg.get("video_gen") if isinstance(cfg, dict) else None
    section = section if isinstance(section, dict) else {}
    own = section.get("uberllm")
    return own if isinstance(own, dict) else {}


def _base_url() -> str:
    return (
        str(_cfg().get("base_url") or os.environ.get("UBERLLM_VIDEO_BASE_URL") or DEFAULT_BASE_URL)
        .strip()
        .rstrip("/")
    )


def _api_key() -> str:
    key_env = str(_cfg().get("key_env") or DEFAULT_KEY_ENV).strip() or DEFAULT_KEY_ENV
    return (get_secret_str(key_env) or "").strip()


_MAX_INLINE_IMAGE_BYTES = 12 * 1024 * 1024
_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif",
}


def _is_remote(value: str) -> bool:
    return str(value or "").strip().lower().startswith(("http://", "https://"))


def _data_uri_from_bytes(data: bytes, name: str) -> str:
    mime = _MIME_BY_SUFFIX.get(os.path.splitext(str(name).split("?")[0])[1].lower(), "image/jpeg")
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def _image_ref(value: str) -> str:
    """Inline a local image as a data URI; pass https/data URIs through.

    UberLLM accepts ``https://`` links or ``data:`` URIs only, so a bare filesystem path
    (what a caller handing over a local file produces) would be rejected as unfetchable.
    """
    v = str(value or "").strip()
    if v.lower().startswith(("http://", "https://", "data:")):
        return v
    path = os.path.expanduser(v.removeprefix("file://"))
    with open(path, "rb") as fh:
        return _data_uri_from_bytes(fh.read(), path)


def _family(model_id: str) -> str:
    for suffix in _SUFFIXES:
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def _configured_model() -> str:
    """``video_gen.model`` as configured (the family/id the caller asked for)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception:
        return ""
    video = cfg.get("video_gen") if isinstance(cfg, dict) else None
    return str(video.get("model") or "").strip() if isinstance(video, dict) else ""


def _enum_of(spec: Any) -> list[Any]:
    return list(spec.get("enum") or []) if isinstance(spec, dict) else []


class UberLLMVideoGenProvider(VideoGenProvider):
    """Text-, image- and reference-to-video through UberLLM's job API."""

    name = "uberllm"
    display_name = "UberLLM"

    def __init__(self) -> None:
        self._catalog_cache: Optional[tuple[list[dict[str, Any]], float]] = None
        self._detail_cache: dict[str, tuple[Optional[dict[str, Any]], float]] = {}

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
                params={"type": "video"},
                headers={"Authorization": f"Bearer {_api_key()}"},
                timeout=_REQUEST_TIMEOUT_S,
            )
            response.raise_for_status()
            data = response.json().get("data")
            entries = [e for e in (data if isinstance(data, list) else []) if isinstance(e, dict) and e.get("id")]
        except Exception as exc:
            logger.debug("UberLLM video catalog unavailable: %s", exc)
        self._catalog_cache = (entries, time.monotonic())
        return entries

    def _model_detail(self, model_id: str) -> Optional[dict[str, Any]]:
        """The model's declared ``parameters.properties`` map, or ``None`` if unknown.

        ``GET /v1/models?type=video`` omits ``parameters``; only the per-model detail
        route lists them, so this is the single source of truth for what a model accepts.
        ``{}`` means "the model declares no optional parameters"; ``None`` (network
        failure) means "unknown — send the caller's values as they are".
        """
        if not model_id:
            return None
        cached = self._detail_cache.get(model_id)
        if cached and time.monotonic() - cached[1] < _CATALOG_TTL_S:
            return cached[0]
        props: Optional[dict[str, Any]] = None
        try:
            import requests

            response = requests.get(
                f"{_base_url()}/models/{model_id}",
                headers=self._headers(),
                timeout=_REQUEST_TIMEOUT_S,
            )
            if response.status_code < 400:
                data = response.json()
                detail = data.get("data") if isinstance(data, dict) else None
                detail = detail if isinstance(detail, dict) else (data if isinstance(data, dict) else {})
                found = (detail.get("parameters") or {}).get("properties")
                props = found if isinstance(found, dict) else {}
        except Exception as exc:
            logger.debug("UberLLM video: model detail for %s unavailable: %s", model_id, exc)
        self._detail_cache[model_id] = (props, time.monotonic())
        return props

    def _configured_detail(self) -> Optional[dict[str, Any]]:
        """Declared parameters of the id the configured family uses for a text call."""
        family = _configured_model()
        if not family:
            return None
        try:
            model_id = self._resolve_model(family, image_url=None, references=[])
        except ValueError:
            return None
        return self._model_detail(model_id)

    def list_models(self) -> List[Dict[str, Any]]:
        """One entry per model family, with the modalities its ids cover."""
        families: dict[str, dict[str, Any]] = {}
        for entry in self._catalog():
            if entry.get("available") is False:
                continue
            model_id = str(entry["id"])
            fam = families.setdefault(
                _family(model_id),
                {"id": _family(model_id), "display": _family(model_id).split("/")[-1], "modalities": set(), "price": ""},
            )
            if model_id.endswith("-image-to-video"):
                fam["modalities"].add("image")
            elif model_id.endswith("-reference-to-video"):
                fam["modalities"].update({"image", "reference"})
            else:
                fam["modalities"].add("text")
            pricing = entry.get("pricing") or {}
            if pricing.get("usd") is not None and not fam["price"]:
                fam["price"] = f"${pricing['usd']}/{pricing.get('unit') or 'unit'}"
        out: list[dict[str, Any]] = []
        for fam in families.values():
            modalities = sorted(fam["modalities"])
            out.append(
                {
                    "id": fam["id"],
                    "display": fam["display"],
                    "modalities": modalities,
                    "strengths": ", ".join(modalities) + "-to-video",
                    "price": fam["price"],
                }
            )
        return sorted(out, key=lambda m: m["id"])

    def default_model(self) -> Optional[str]:
        return None

    def capabilities(self) -> dict[str, Any]:
        caps: dict[str, Any] = {
            "modalities": ["text", "image"],
            "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"],
            "resolutions": ["480p", "720p", "1080p"],
            "max_duration": 15,
            "min_duration": 1,
            "supports_audio": False,
            "supports_negative_prompt": False,
            "supports_seed": False,
            "supports_upscale": False,
            "max_reference_images": 4,
        }
        # The tool schema is built from this, so advertise what the configured model
        # actually declares. Ground truth beats a shared list: ``uncensored-t2v-turbo``
        # accepts 480p/720p and no ``aspect_ratio`` at all, and UberLLM silently drops
        # fields a model does not know rather than rejecting them.
        props = self._configured_detail() or {}
        aspect_ratios, resolutions = _enum_of(props.get("aspect_ratio")), _enum_of(props.get("resolution"))
        durations = _enum_of(props.get("duration_seconds"))
        if aspect_ratios:
            caps["aspect_ratios"] = [str(v) for v in aspect_ratios]
        if resolutions:
            caps["resolutions"] = [str(v) for v in resolutions]
        numeric = [d for d in durations if isinstance(d, (int, float)) and not isinstance(d, bool)]
        if numeric:
            caps["min_duration"], caps["max_duration"] = int(min(numeric)), int(max(numeric))
        return caps

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "UberLLM",
            "badge": "paid",
            "tag": "Veo, Kling, Seedance, Wan, LTX … — live catalog from api.uberllm.dev; text-, image- and reference-to-video",
            "env_vars": [
                {
                    "key": DEFAULT_KEY_ENV,
                    "prompt": "UberLLM API key",
                    "url": "https://uberllm.dev/dashboard/keys",
                }
            ],
        }

    # ---- model routing -------------------------------------------------------------------------
    def _resolve_model(self, configured: Optional[str], *, image_url: Optional[str], references: list[str]) -> str:
        """Map the configured family/id to the id that matches this call's modality."""
        wanted = "-reference-to-video" if references else ("-image-to-video" if image_url else "-text-to-video")
        available = {str(e["id"]) for e in self._catalog() if e.get("available") is not False}
        base = _family(configured) if configured else ""
        if not base:
            raise ValueError(
                "No UberLLM video model configured. Set one with `hermes config set video_gen.model <id>` "
                "(see `hermes tools` → Video Generation)."
            )
        candidate = base + wanted
        candidates = [candidate]
        # fal-style ids keep the modality as its own path segment ("…/standard/text-to-video"),
        # so the wanted suffix must REPLACE that segment instead of being appended to the whole
        # id. Appending produced "…/standard-image-to-video", a nonexistent id, and the call then
        # fell back to the configured text-to-video model with the image silently ignored.
        head = str(configured or "").rsplit("/", 1)
        if len(head) == 2 and head[1] in {s.lstrip("-") for s in _SUFFIXES}:
            slash_form = head[0] + "/" + wanted.lstrip("-")
            if slash_form not in candidates:
                candidates.append(slash_form)
        elif str(configured or "").strip():
            # A family directory without a modality segment — what the README tells you to
            # configure: "fal-ai/minimax/hailuo-02/standard" → "…/standard/text-to-video".
            dir_form = str(configured).strip().rstrip("/") + "/" + wanted.lstrip("-")
            if dir_form not in candidates:
                candidates.append(dir_form)
        for candidate in candidates:
            if candidate in available:
                return candidate
        # Only fall back to the configured id when it already has the modality this call needs;
        # otherwise the request would run with the wrong modality (image dropped).
        if configured in available and str(configured).endswith(wanted):
            return str(configured)
        # Some ids name their modality instead of suffixing it
        # ("uncensored/uncensored-t2v-turbo", "kling/kling-3-0-turbo", "fal-ai/veo3.1").
        # For those the family+suffix dance above has nothing to build, and rejecting the
        # call made every such model unusable. Accept the configured id verbatim when it
        # is available and its own schema covers this call: an image call needs a declared
        # image input, a text call needs a known (non image-only) model.
        if str(configured or "") in available:
            detail = self._model_detail(str(configured))
            if references:
                fits = isinstance((detail or {}).get("input_images"), dict)
            elif image_url:
                fits = isinstance((detail or {}).get("image_url"), dict)
            else:
                fits = detail is not None
            if fits:
                return str(configured)
        raise ValueError(
            f"UberLLM has no available model for this call: tried {candidates!r}. "
            "Pick another family with `hermes config set video_gen.model <id>`."
        )

    # ---- generation ----------------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[list[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        resolution: str = "720p",
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raw_references = [u for u in (reference_image_urls or []) if isinstance(u, str) and u.strip()]
        raw_image = image_url.strip() if isinstance(image_url, str) and image_url.strip() else None
        try:
            model_id = self._resolve_model(model, image_url=raw_image, references=raw_references)
        except ValueError as exc:
            return error_response(error=str(exc), error_type="config_error", provider=self.name,
                                  model=model or "", prompt=prompt, aspect_ratio=aspect_ratio)
        if not _api_key():
            return error_response(error="No UberLLM API key: set HERMES_CUSTOM_API_UBERLLM_DEV_API_KEY",
                                  error_type="auth_error", provider=self.name, model=model_id,
                                  prompt=prompt, aspect_ratio=aspect_ratio)

        session = self._session()
        try:
            references = [self._inline_image(session, u) for u in raw_references]
            primary_image = self._inline_image(session, raw_image) if raw_image else None
        except OSError as exc:
            session.close()
            return error_response(error=f"Cannot read the source image: {exc}", error_type="invalid_request",
                                  provider=self.name, model=model_id, prompt=prompt,
                                  aspect_ratio=aspect_ratio)

        payload: dict[str, Any] = {"model": model_id, "prompt": prompt}
        declared = self._model_detail(model_id)
        ignored: list[dict[str, Any]] = []
        applied: dict[str, Any] = {}

        def _send(key: str, value: Any) -> None:
            """Send a parameter only when the model declares it and accepts the value.

            UberLLM tolerates unknown fields and drops them without an error, so an
            unsupported ``aspect_ratio`` used to look applied while the provider chose
            the frame size itself (840×480 for a 16:9 request), and ``resolution`` was
            never sent at all. Omitted values are reported in ``ignored_params``.
            """
            if value is None or value == "" or value == 0 or value is False:
                return
            spec = (declared or {}).get(key) if declared is not None else None
            if declared is not None and not isinstance(spec, dict):
                ignored.append({"param": key, "value": value, "reason": "not supported by this model"})
                return
            allowed = _enum_of(spec)
            if allowed and value not in allowed:
                ignored.append({"param": key, "value": value,
                                "reason": "allowed: " + ", ".join(str(a) for a in allowed)})
                return
            payload[key] = value
            applied[key] = value

        _send("duration_seconds", int(duration) if duration else None)
        _send("aspect_ratio", aspect_ratio)
        _send("resolution", resolution)
        if references:
            payload["input_images"] = [{"url": u, "role": "reference"} for u in references]
        elif primary_image:
            payload["image_url"] = primary_image
        max_cost = _cfg().get("max_cost")
        if max_cost:
            payload["max_cost"] = float(max_cost)

        modality = "image" if (references or primary_image) else "text"
        try:
            submitted = session.post(
                f"{_base_url()}/video/generations",
                headers={**self._headers(), "Idempotency-Key": str(uuid.uuid4())},
                json=payload,
                timeout=_REQUEST_TIMEOUT_S,
            )
            if submitted.status_code >= 400:
                return error_response(error=_error_text(submitted), error_type="provider_error",
                                      provider=self.name, model=model_id, prompt=prompt,
                                      aspect_ratio=aspect_ratio)
            job = submitted.json()
            job_id = str(job.get("id") or "")
            if not job_id:
                return error_response(error=f"UberLLM returned no job id: {job}", error_type="provider_error",
                                      provider=self.name, model=model_id, prompt=prompt,
                                      aspect_ratio=aspect_ratio)
            finished = self._poll(session, job_id)
            if finished.get("status") != "succeeded":
                return error_response(
                    error=f"UberLLM job {job_id} ended {finished.get('status')}: "
                          f"{(finished.get('error') or {}).get('message') or finished.get('error') or 'no detail'}",
                    error_type="provider_error", provider=self.name, model=model_id,
                    prompt=prompt, aspect_ratio=aspect_ratio)
            results = finished.get("results") or []
            url = str((results[0] or {}).get("url") or "") if results else ""
            if not url:
                return error_response(error=f"UberLLM job {job_id} succeeded without a video URL",
                                      error_type="provider_error", provider=self.name, model=model_id,
                                      prompt=prompt, aspect_ratio=aspect_ratio)
            try:
                video = str(save_url_video(url, prefix="uberllm", headers={"Authorization": f"Bearer {_api_key()}"},
                                           trusted_origin=True))
            except Exception as exc:  # keep the (short-lived) URL rather than losing the result
                logger.warning("UberLLM video download failed (%s); returning the artifact URL", exc)
                video = url
            usage = finished.get("usage") or {}
            # The charge lands a beat after status flips to succeeded: a real 480p job
            # reported cost_usd=null (estimate 0.102) while the job detail answered
            # usage.cost_usd=0.51 seconds later. The estimate is not the bill, so re-read
            # the job a few times instead of reporting null.
            for _ in range(3):
                if usage.get("cost_usd") is not None:
                    break
                time.sleep(2)
                try:
                    usage = self._poll(session, job_id).get("usage") or usage
                except Exception:
                    break
            return success_response(
                video=video, model=model_id, prompt=prompt, modality=modality,
                aspect_ratio=str(applied.get("aspect_ratio") or ""),
                duration=int(applied.get("duration_seconds") or 0), provider=self.name,
                extra={"job_id": job_id, "cost_usd": usage.get("cost_usd"),
                       "estimated_cost_usd": finished.get("estimated_cost_usd"),
                       "resolution": str(applied.get("resolution") or "provider default"),
                       "applied": applied, "ignored_params": ignored},
            )
        except Exception as exc:
            return error_response(error=f"UberLLM video request failed: {exc}", error_type="provider_error",
                                  provider=self.name, model=model_id, prompt=prompt,
                                  aspect_ratio=aspect_ratio)
        finally:
            session.close()

    def _inline_image(self, session: Any, value: str) -> str:
        """Local paths and remote URLs both become data URIs.

        UberLLM takes ``https://`` links or ``data:`` URIs only, and when a provider fetches
        the link itself it fails the whole job on hosts that block server-side requests
        ("The video could not be made. Nothing was charged." — wikimedia.org does that).
        Inlining with our own download removes the external fetch; if that download fails the
        URL is passed through so the provider can still try.
        """
        v = str(value or "").strip()
        if v.lower().startswith("data:"):
            return v
        if _is_remote(v):
            try:
                response = session.get(
                    v, timeout=_REQUEST_TIMEOUT_S, allow_redirects=True,
                    headers={"User-Agent": "hermes-agent/1.0 (uberllm video backend)"},
                )
                size = len(response.content or b"")
                if response.status_code < 400 and 0 < size <= _MAX_INLINE_IMAGE_BYTES:
                    return _data_uri_from_bytes(response.content, v)
                logger.warning("UberLLM: cannot inline %s (HTTP %s, %s bytes); passing the URL through",
                               v[:100], response.status_code, size)
            except Exception as exc:
                logger.warning("UberLLM: download of %s failed (%s); passing the URL through", v[:100], exc)
            return v
        return _image_ref(v)

    def _poll(self, session: Any, job_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + _POLL_DEADLINE_S
        while True:
            response = session.get(f"{_base_url()}/video/generations/{job_id}",
                                   headers=self._headers(), timeout=_REQUEST_TIMEOUT_S)
            if response.status_code >= 400:
                raise RuntimeError(_error_text(response))
            job = response.json()
            if job.get("status") in {"succeeded", "failed", "canceled", "expired"}:
                return job
            if time.monotonic() > deadline:
                raise TimeoutError(f"UberLLM job {job_id} still {job.get('status')} after {_POLL_DEADLINE_S:.0f}s")
            time.sleep(_POLL_INTERVAL_S)


def _error_text(response: Any) -> str:
    try:
        body = response.json()
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            return f"HTTP {response.status_code}: {err.get('message') or err}"
        return f"HTTP {response.status_code}: {body}"
    except Exception:
        return f"HTTP {response.status_code}: {response.text[:300]}"


def register(ctx) -> None:
    """Plugin entry point — wire the UberLLM video and image backends into the registries."""
    ctx.register_video_gen_provider(UberLLMVideoGenProvider())
    try:
        from .image_backend import UberLLMImageGenProvider
    except Exception as exc:  # an image-side import error must not take video down with it
        logger.warning("UberLLM image backend unavailable: %s", exc)
    else:
        ctx.register_image_gen_provider(UberLLMImageGenProvider())
