# Subtitlegen

Local, resumable subtitle generation with word-level timing. See the [design and requirements](docs/README.md).

## Mac Apple Silicon

Requires Python 3.11 or 3.12. FFmpeg is optional because media access uses PyAV.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[mac,ocr]'
.venv/bin/subtitlegen generate /path/to/videos --preset quality
```

`auto` selects MLX when its optional package is installed and falls back to faster-whisper CPU otherwise.

## Windows with NVIDIA Docker

Install Docker Desktop, NVIDIA drivers, and NVIDIA Container Toolkit. Copy `.env.example` to `.env` and set **host** paths there. Use forward slashes on Windows (`C:/Users/...`). Do not pass Windows paths to `generate`; inside the container videos are always `/data/videos`.

```powershell
Copy-Item .env.example .env
# edit VIDEO_HOST_PATH, then:
docker compose --profile ocr run --rm visual
```

That command reads `SUBTITLEGEN_PRESET`, cache, and OCR probe settings from `.env`. Generated SRT (dialogue) and ASS (dialogue + on-screen titles) are written beside the mounted videos.

Other images, if you need them:

```powershell
docker compose run --rm subtitler
docker compose --profile whisperx run --rm whisperx
docker compose --profile nemo run --rm parakeet
```

Existing `.srt` files are skipped. Pass `--overwrite` or set `SUBTITLEGEN_OVERWRITE=1` only when you want to regenerate them.

## Commands

```bash
subtitlegen generate VIDEO_OR_DIRECTORY
subtitlegen generate VIDEO_OR_DIRECTORY --preset fast|quality|english-fast
subtitlegen generate VIDEO --profile one-piece --arc Dressrosa
subtitlegen generate VIDEO --no-visual-text
subtitlegen validate SUBTITLE.srt
subtitlegen benchmark VIDEO [--backend auto]
```

`--preset quality` uses Whisper `large-v3` (WhisperX forced alignment when that package is installed; otherwise faster-whisper large-v3). `--preset fast` is turbo and is weaker on names and grammar.

Give a file or directory. The CLI infers the series name, expands a glossary from shipped YAML plus Wikipedia/search, applies gated spelling correction, and runs Japanese on-screen translation for anime. OCR first scans coarsely for Japanese characters, then reads only the windows around hits.

Configuration defaults are in `config.ini`. Docker defaults are in `.env`. First use may download a model; cached runs can operate offline.
See [ASR backends and presets](docs/07-asr-backends.md) for optional CUDA adapters.
See [Japanese visual text](docs/08-visual-text.md) for local OCR, translation, and ASS output.
For clean Mac/Windows setup and the RTX acceptance runner, see
[platform setup](docs/09-platform-setup.md). Review the
[model/license manifest](docs/10-model-licenses.md), especially NLLB's
non-commercial restriction, before deployment.
