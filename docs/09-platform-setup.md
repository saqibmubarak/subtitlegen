# Platform setup and acceptance

## macOS Apple Silicon

Prerequisites: macOS on Apple Silicon, Python 3.11 or 3.12, and enough free
storage for model caches. FFmpeg is optional.

```bash
git clone <repository-url>
cd subtitlegen
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[mac,ocr,dev]'
ruff check .
mypy src main.py config.py transcriber.py
pytest -q
subtitlegen --help
```

Models download on first use. To keep all caches under the checkout:

```bash
export HF_HOME="$PWD/model_cache/huggingface"
export PADDLE_PDX_CACHE_HOME="$PWD/model_cache/paddle"
subtitlegen generate "/path/with spaces/episode.mp4" \
  --preset quality
```

The dialogue-only SRT and combined ASS are written beside the video unless
`--output-dir` is supplied.

## Windows with RTX 3070 Ti

Prerequisites:

1. Windows 11 with current NVIDIA drivers.
2. Docker Desktop using the WSL2 backend.
3. NVIDIA Container Toolkit configured so `docker run --gpus all` works.
4. Git and PowerShell 7.

From a clean checkout, copy `.env.example` to `.env` and use absolute Windows
paths. Docker Desktop converts these mounted paths into Linux container paths.

```powershell
Copy-Item .env.example .env
# Set VIDEO_HOST_PATH with forward slashes, then:
docker compose config --quiet
docker compose --profile windows build
docker compose --profile windows up
```

Generate options (`SUBTITLEGEN_PRESET`, cache, OCR probe interval) come from
`.env`. Existing SRT files are skipped unless `SUBTITLEGEN_OVERWRITE=1`.

The repeatable acceptance script builds all runtime and test images, runs the
unit/integration suite in each dependency profile, verifies CUDA visibility and
the Compose launch path, reruns a batch for resume/skip, corrupts and recovers
one cached ASR artifact, runs required Avatar and Dressrosa samples, and verifies
explicit CPU fallback:

```powershell
pwsh ./scripts/windows-rtx-acceptance.ps1 `
  -VideoPath "D:\Anime With Spaces" `
  -AvatarFile "Avatar Episode.mp4" `
  -DressrosaFile "[Muhn Pace] Dressrosa 03.mp4"
```

Record console output, driver/container versions, wall time, peak VRAM from
`nvidia-smi`, and playback review in
`docs/benchmarks/phase-5-windows-rtx.md`. Do not tune batch sizes from Mac
results; change the 8 GB presets only from this measured run.

## Failure recovery

- A completed stage resumes only when its artifact passes validation.
- Failed stages retain an actionable error and rerun on the next invocation.
- Ctrl+C marks the active stage as cancelled and releases the one-GPU token.
- Manifest schema 1 is migrated to schema 2 when loaded.
- Delete only the affected job directory under the configured cache to force a
  clean job; model caches are independent and do not need to be removed.

See the [model and license manifest](10-model-licenses.md) before distributing
models or generated services.
