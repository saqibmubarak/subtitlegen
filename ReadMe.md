# Subtitlegen

Local, resumable subtitle generation with word-level timing. See the [design and requirements](docs/README.md).

## Mac Apple Silicon

Requires Python 3.11 or 3.12. FFmpeg is optional because media access uses PyAV.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[mac]'
.venv/bin/subtitlegen generate /path/to/videos --backend auto
```

`auto` selects MLX when its optional package is installed and falls back to faster-whisper CPU otherwise.

## Windows with NVIDIA Docker

Install Docker Desktop, NVIDIA drivers, and NVIDIA Container Toolkit. Copy `.env.example` to `.env`, set the host paths, then run:

```bash
docker compose run --rm subtitler
```

The image uses one GPU, a persistent Hugging Face model cache, a resumable job cache, and unbuffered logs. Generated SRT files are written beside mounted videos.

## Commands

```bash
subtitlegen generate VIDEO_OR_DIRECTORY [--backend auto] [--overwrite]
subtitlegen generate VIDEO --profile one-piece --arc Dressrosa
subtitlegen validate SUBTITLE.srt
subtitlegen benchmark VIDEO [--backend auto]
```

Configuration defaults are in `config.ini`. First use may download a model; cached runs can operate offline.
