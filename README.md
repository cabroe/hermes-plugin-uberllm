# hermes-plugin-uberllm

A **video and image generation backend for [Hermes](https://github.com/NousResearch/hermes-agent)**
that speaks [UberLLM](https://api.uberllm.dev). It gives Hermes' built-in `video_generate` and
`image_generate` tools a working UberLLM backend without touching Hermes core.

Works with every model UberLLM lists under `GET /v1/models?type=video` and `?type=image` — Hailuo,
Kling, Seedance, Veo, LTX, Vidu, Wan, Grok Imagine, Flux, Nano Banana, Ideogram, Krea — and follows
the catalog live, so retired models drop out by themselves.

## Why this exists

Two things the built-in backends cannot do:

| Hermes tool | Bundled backends | This plugin |
|---|---|---|
| `video_generate` | fal, deepinfra, openrouter, xai — none speaks UberLLM's async `POST /v1/video/generations` → poll → download | Full job lifecycle, per-modality model ids, image inputs inlined as `data:` URIs |
| `image_generate` | the openai plugin asks for `url` output and then downloads the file **without** the `Authorization` header → 401, so the image never lands in the cache | Asks for `response_format: b64_json` and saves the bytes directly |

## Install

```bash
# requires a GitHub reachable from the machine (public repo, no login needed)
hermes plugins install cabroe/hermes-plugin-uberllm
```

Then point Hermes at it:

```bash
hermes plugins enable uberllm
hermes config set video_gen.provider uberllm
hermes config set video_gen.model "fal-ai/minimax/hailuo-02/standard/text-to-video"
hermes config set image_gen.provider uberllm
hermes config set image_gen.uberllm.model "venice/z-image-turbo"
```

Put the API key in `~/.hermes/.env` (never in `config.yaml`):

```bash
HERMES_CUSTOM_API_UBERLLM_DEV_API_KEY=...
```

Pin a commit for reproducibility: `hermes plugins install cabroe/hermes-plugin-uberllm --ref <40-char-sha>`.

## Video

Just ask Hermes for a video, or call the tool:

- text-to-video: `prompt` only
- image-to-video: add `image_url` (public URL, `data:` URI, or a local file path)
- reference-to-video: several images as references

`video_gen.model` is a **family** id. The backend strips the modality suffix and re-applies the one
the call needs, so one configured family serves all three modalities. A directory-style family
(`fal-ai/minimax/hailuo-02/standard`), a full id (`…/standard/text-to-video`) and an id that names its
modality instead of suffixing it (`uncensored/uncensored-t2v-turbo`, `fal-ai/veo3.1`) all resolve.

### What it handles for you

| UberLLM quirk | What the plugin does |
|---|---|
| Model ids carry the modality as a path segment (`…/standard/text-to-video`) | Replaces the trailing segment instead of appending to the whole id, and refuses to fall back to a model of the wrong modality |
| Images must be `https://` or `data:`; the provider fetches remote links itself and fails the job on hosts that block server-side requests | Downloads remote images and inlines them as `data:` URIs; local paths and `file://` are base64'd |
| Async jobs (`POST` → `202 {id}` → poll) | Polls every 5 s up to 15 min, sends `Idempotency-Key`, downloads the artifact with `Authorization` |
| Catalog mixes available and unavailable models | Filters `available: false` out of candidates |
| `GET /v1/models?type=video` omits `parameters`; **unknown fields are dropped, not rejected** | Reads `GET /v1/models/<id>` per model (10 min cache) and sends only the parameters that model declares, with a value from its own enum. Values it cannot honour come back in `ignored_params` instead of looking applied |

## Image

`image_gen.uberllm.model` takes a concrete id from `GET /v1/models?type=image`:

- text-to-image: any `/v1/images/generations` model (`venice/z-image-turbo`, `fal-ai/flux/schnell`, …)
- editing: an `…-edit` model that lists `/v1/images/edits` and `image` as an input modality
  (`bytedance/seedream-v4-edit`, `google/nano-banana-2-1-edit`, …)

`image_url` / `reference_image_urls` route to `/v1/images/edits`; local files are inlined as data
URIs. The tool's advertised modalities follow the configured model, so a text-only model does not
advertise editing it cannot do — and asking anyway returns a `config_error` naming a cheap edit model
to switch to.

| UberLLM quirk | What the plugin does |
|---|---|
| Default `response_format` is `url`, and that file link needs the API key | Requests `response_format: b64_json`; falls back to fetching the URL **with** the auth header |
| `size` is a per-model enum, and the list route does not include it | Reads `GET /v1/models/<id>` once per model (10 min cache) and picks the closest size for the semantic aspect (`1536x1024` landscape, `1024x1024` square, `1024x1536` portrait) |
| Text-only and edit-capable models are different ids | Routes by the model's own `endpoints` / `input_modalities` instead of guessing from the name |

## Things worth knowing (measured, not guessed)

- **Duration is validated per model.** `fal-ai/minimax/hailuo-02/standard/*` makes 6 s or 10 s clips
  only; asking for 3 s returns HTTP 400 and the provider's text comes through verbatim.
- **`resolution` is a per-model parameter, and it has to be sent explicitly.** Models that list it
  (`uncensored/uncensored-t2v-turbo`: `480p`, `720p`) use the provider default when it is omitted;
  models that do not list it ignore the field entirely (`fal-ai/minimax/hailuo-02/standard/*`). The
  plugin now sends it whenever the model declares it, and reports the value it did **not** send in
  `ignored_params` (`{"param": "resolution", "value": "1080p", "reason": "allowed: 480p, 720p"}`).
- **`aspect_ratio` is not honoured by every model, and the output size is not a promise.**
  `uncensored/uncensored-t2v-turbo` declares no `aspect_ratio` at all and returns 840×480 (1.75:1) for
  a 16:9 request; 27 of the 100 available video models do declare it. Read `resolution` /
  `ignored_params` in the tool result, and `ffprobe` the file, before assuming a frame size.
- **Measured 480p job** (`uncensored/uncensored-t2v-turbo`, 5 s prompt, 2026-11-09): 840×480, 30 fps,
  161 frames, 5.37 s, 512 KB. Billed **$0.51** while `estimated_cost_usd` said $0.102 — the estimate
  is off by 5× on this model, so read `cost_usd`. `usage` also lands a few seconds *after* the job
  reports `succeeded`; the plugin re-reads the job up to 3× (2 s apart) instead of returning null.
- **Output follows the source image's aspect** for image-to-video, not `aspect_ratio`: a 4:3 photo
  returns 1024×768 even when 16:9 is requested. A 6 s request yields a 5.875 s file at 24 fps.
- **Listed price is not what you pay.** `pricing.usd` in the catalog says $0.025/s for hailuo
  text-to-video and $0.01445/s for its image-to-video; a 6 s 720p clip billed **$0.2295** in both
  cases. Same on images: `bytedance/seedream-v4-edit` lists $0.025/image and billed $0.03.
  The `cost_usd` in the tool result is the real charge — read that, not the catalog.
- **A failed job charges nothing.** Retry freely while iterating on a prompt or an input image.
  `venice/z-image-turbo` billed $0.00396 for a 1024×1024 image.
- **Verify the output, not the status.** `success: true` plus `modality: image` does not prove the
  source image was honoured — pull the first and last frame of a video with
  `ffmpeg -ss <t> -frames:v 1`, and eyeball a generated image, before trusting it.

## Requirements

- Hermes with `video_generate` / `image_generate` tool support (custom `kind: backend` plugins)
- An UberLLM API key with credit for the modalities you use
- `ffmpeg` only if you want to inspect generated files

No Python dependencies beyond what Hermes ships.

## License

MIT — see [LICENSE](LICENSE).
