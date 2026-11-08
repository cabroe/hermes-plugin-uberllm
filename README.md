# hermes-plugin-uberllm

A **video generation backend for [Hermes](https://github.com/NousResearch/hermes-agent)** that speaks
[UberLLM](https://api.uberllm.dev)'s async job API. It gives Hermes' built-in `video_generate` tool
text-to-video, image-to-video and reference-to-video without touching Hermes core.

Works with every model UberLLM lists under `GET /v1/models?type=video` — Hailuo, Kling, Seedance,
Veo, LTX, Vidu, Wan, Grok Imagine — and follows the catalog live, so retired models drop out by
themselves.

## Install

```bash
# requires a GitHub reachable from the machine (public repo, no login needed)
hermes plugins install cabroe/hermes-plugin-uberllm
```

Then point Hermes at it and pick a model family:

```bash
hermes plugins enable uberllm
hermes config set video_gen.provider uberllm
hermes config set video_gen.model "fal-ai/minimax/hailuo-02/standard/text-to-video"
```

Put the API key in `~/.hermes/.env` (never in `config.yaml`):

```bash
HERMES_CUSTOM_API_UBERLLM_DEV_API_KEY=...
```

Pin a commit for reproducibility: `hermes plugins install cabroe/hermes-plugin-uberllm --ref <40-char-sha>`.

## Usage

Just ask Hermes for a video, or call the tool:

- text-to-video: `prompt` only
- image-to-video: add `image_url` (public URL, `data:` URI, or a local file path)
- reference-to-video: several images as references

`video_gen.model` is a **family** id. The backend strips the modality suffix and re-applies the one
the call needs, so one configured family serves all three modalities.

## What this backend handles for you

| UberLLM quirk | What the plugin does |
|---|---|
| Model ids carry the modality as a path segment (`…/standard/text-to-video`) | Replaces the trailing segment instead of appending to the whole id, and refuses to fall back to a model of the wrong modality |
| Images must be `https://` or `data:`; the provider fetches remote links itself and fails the job on hosts that block server-side requests | Downloads remote images and inlines them as `data:` URIs; local paths and `file://` are base64'd |
| Async jobs (`POST` → `202 {id}` → poll) | Polls every 5 s up to 15 min, sends `Idempotency-Key`, downloads the artifact with `Authorization` |
| Catalog mixes available and unavailable models | Filters `available: false` out of candidates |

## Things worth knowing (measured, not guessed)

- **Duration is validated per model.** `fal-ai/minimax/hailuo-02/standard/*` makes 6 s or 10 s clips
  only; asking for 3 s returns HTTP 400 and the provider's text comes through verbatim.
- **Output follows the source image's aspect** for image-to-video, not `aspect_ratio`: a 4:3 photo
  returns 1024×768 even when 16:9 is requested. A 6 s request yields a 5.875 s file at 24 fps.
- **Listed price is not what you pay.** `pricing.usd` in the catalog says $0.025/s for hailuo
  text-to-video and $0.01445/s for its image-to-video; a 6 s 720p clip billed **$0.2295** in both
  cases. The `cost_usd` in the tool result is the real charge.
- **A failed job charges nothing.** Retry freely while iterating on a prompt or an input image.
- **Verify the image was used.** `success: true` plus `modality: image` does not prove the source
  image was honoured — pull the first and last frame with `ffmpeg -ss <t> -frames:v 1` and compare
  against the input.

## Requirements

- Hermes with a `video_generate` tool (custom `kind: backend` plugin support)
- An UberLLM API key with credit for video generation
- `ffmpeg` only if you want to inspect generated files

No Python dependencies beyond what Hermes ships.

## License

MIT — see [LICENSE](LICENSE).