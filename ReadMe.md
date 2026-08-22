# Subtitlegen

Local subtitle generator for dubbed video. It transcribes English dialogue with
word-level timing, corrects series names from a glossary, and (for anime)
translates Japanese on-screen titles into a second ASS layer.

You run it on a file or a folder. It writes:

| File | What it is |
|---|---|
| `episode.srt` | Dialogue only (players, editors, compatibility) |
| `episode.ass` | Dialogue plus yellow `OnScreen` titles |
| `.subtitlegen/` | Resume cache (ASR, jobs, models if you point caches here) |

It does **not** upload video. Models download on first use, then work offline.

Python **3.11 or 3.12**. FFmpeg is optional; media I/O uses PyAV. Design notes
live under [docs/](docs/README.md).

---

## What a run does

1. **Discover** `.mp4` / `.mkv` / `.avi` / `.mov` / `.wmv` (see `config.ini`).
2. **Infer a series profile** from the path (`Dressrosa` → One Piece). Expand
   names from shipped YAML plus optional Wikipedia (`SUBTITLEGEN_ENRICH_GLOSSARY`).
3. **ASR** with the selected preset. Existing valid `.srt` files are skipped
   unless you pass `--overwrite`.
4. **Cue building** (max length, gaps, punctuation) and glossary spelling fix
   (`Doflamingo`, not `Dofuramingo`).
5. **On-screen Japanese** (default on): coarse 4 s probe for title-script
   (kanji / katakana), then 1 s refine in a ±12 s window. Paddle OCR reads HUD
   cards; NLLB-200 translates to English. Use `--no-visual-text` to skip.

`--preset` only picks the **speech** model. Visual OCR is the same on every
preset.

---

## Presets

`--preset` / `SUBTITLEGEN_PRESET` chooses the dialogue backend. `auto` backend
picks the adapter from what is installed.

| Preset | Apple Silicon (MLX installed) | NVIDIA CUDA | CPU fallback |
|---|---|---|---|
| **`quality`** (default in Docker) | MLX Whisper **large-v3** | WhisperX large-v3 if the WhisperX image/package is present, else faster-whisper large-v3 fp16 | faster-whisper large-v3 int8 |
| **`fast`** | MLX **large-v3-turbo** | faster-whisper turbo fp16 | faster-whisper turbo int8 |
| **`english-fast`** | same as `fast` (Parakeet is CUDA-only) | NVIDIA Parakeet TDT 0.6B v3 | same as `fast` |

`quality` is slower and better on names. `fast` is weaker on grammar and
glossary terms. `english-fast` is English-only and needs the NeMo image.

Override the adapter with `--backend auto|mlx|faster-whisper|whisperx|parakeet`.

---

## macOS Apple Silicon (native)

From a clone of this repo. First run downloads several GB of models (Whisper,
Paddle, NLLB) and needs a network connection.

```bash
cd subtitlegen
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[mac,ocr]'
subtitlegen --help

subtitlegen generate "/path/to/episode.mp4" --preset quality
subtitlegen generate "/path/to/folder" --preset quality --overwrite
```

Keep caches inside the repo if you want:

```bash
export HF_HOME="$PWD/.subtitlegen/hf"
export PADDLE_PDX_CACHE_HOME="$PWD/.subtitlegen/paddle"
subtitlegen generate "samples/[Muhn Pace] Dressrosa 01.mp4" \
  --preset quality --cache-dir .subtitlegen
```

Docker on Mac works but has no NVIDIA GPU. Prefer native. If you must use
Compose, drop GPU reservations:

```bash
docker compose -f docker-compose.yml -f docker-compose.mac.yml --profile ocr run --rm visual
```

Parakeet (`english-fast` / `parakeet` service) cannot run on Mac.

---

## Windows (Docker + NVIDIA) — recommended

Native Windows + PaddleOCR is brittle. Use **Docker Desktop (WSL2)** plus
current NVIDIA drivers so `docker run --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` works.

Prerequisites: Windows 11, Git, PowerShell, Docker Desktop with the WSL2
backend and GPU enabled.

### 1. Clone, copy `.env`, point at your videos

```powershell
cd subtitlegen
Copy-Item .env.example .env
notepad .env
```

Set `VIDEO_HOST_PATH` to the folder that **already contains** the `.mp4` /
`.mkv` files. The default `./videos` is a folder next to `docker-compose.yml`;
create it and copy videos there, or set an absolute path.

Use **forward slashes** (`C:/Users/you/Videos/Dressrosa`). A backslash before
`n` becomes a newline and the mount will be wrong.

| Variable | Purpose | Typical value |
|---|---|---|
| `VIDEO_HOST_PATH` | Host folder of videos (required) | `C:/Users/you/Videos/Dressrosa` |
| `MODEL_CACHE_HOST_PATH` | Hugging Face / Paddle weights | `./model_cache` |
| `JOB_CACHE_HOST_PATH` | Resume jobs | `./.subtitlegen-docker` |
| `SUBTITLEGEN_PRESET` | `quality`, `fast`, or `english-fast` | `quality` |
| `SUBTITLEGEN_BACKEND` | Usually `auto` | `auto` |
| `SUBTITLEGEN_CACHE_DIR` | Cache **inside** the container | `/cache` |
| `SUBTITLEGEN_PROFILE` | Not forwarded from `.env` (empty string would break inference). Pass a flag. | `--profile one-piece` |
| `SUBTITLEGEN_ARC` | Same as profile | `--arc Dressrosa` |
| `SUBTITLEGEN_VISUAL_PROBE_SECONDS` | Coarse title scan | `4` |
| `SUBTITLEGEN_VISUAL_REFINE_SECONDS` | Window around a hit | `12` |
| `SUBTITLEGEN_ENRICH_GLOSSARY` | Wikipedia name fetch | `1` |
| `SUBTITLEGEN_OVERWRITE` | Rebuild existing SRT | `0`, or `1` to regenerate |

Videos are always **`/data/videos` inside the container**. Do not pass
`D:\...` as the `generate` path.

### 2. Windows default: one Compose pipeline

Parakeet and Paddle stay in **two images** (one GPU owner each; a combined
image would be huge and would fight for VRAM). Compose runs them **in order**
on the same `VIDEO_HOST_PATH`: dialogue writes `.srt`, titles reuse it and
write `.ass`.

Every service is behind a **profile**. A plain `docker compose up` starts
nothing.

`up` here is **foreground** (stay in the terminal). That is how Compose runs
two services in order (`windows-dialogue`, then `windows-titles`). Do **not**
add `-d` / `--detach` — that background form is for servers that keep running.
`down` is only cleanup: `up` has no `--rm`, so it leaves stopped containers.

```powershell
# First time, after src/ changes, or after changing pyproject extras
docker compose --profile windows build

# Foreground: Parakeet SRT, then OCR ASS (not `up -d`)
docker compose --profile windows up

# Optional: delete the stopped job containers
docker compose --profile windows down
```

`windows build` / `windows up` is the default Windows command. After a Python
change you must **rebuild** — `run` does not copy new code into an old image.
`src/` and `ReadMe.md` rebuilds reinstall only the local package. Third-party
wheels reinstall only when `pyproject.toml` extras change. Hugging Face
weights in `MODEL_CACHE_HOST_PATH` stay on disk.

Parakeet (`english-fast`, backend `auto`) runs a three-stage pipeline: decode
the next files with ffmpeg, keep the model on GPU, and write the previous SRT
on a background thread. Windows are sent in **one batched 20 s transcribe**.
A full-episode tensor OOMs; if a batch does, it splits and retries.

Do **not** set `SUBTITLEGEN_OVERWRITE=1` on the title image. Overwrite would
wipe the Parakeet SRT and re-transcribe with faster-whisper.

| Profile | Command | What it does |
|---|---|---|
| `windows` | `docker compose --profile windows up` | **Recommended.** Parakeet, then OCR |
| `nemo` | `docker compose --profile nemo run --rm parakeet` | Dialogue only. `--rm` deletes the container when it exits |
| `ocr` | `docker compose --profile ocr run --rm visual` | Titles only (reuses SRT), or Whisper + titles |
| `subtitler` | `docker compose --profile subtitler run --rm subtitler generate /data/videos --no-visual-text` | Dialogue only, no NeMo/OCR |
| `whisperx` | `docker compose --profile whisperx run --rm whisperx` | WhisperX alignment |

Rebuild one image:

```powershell
docker compose --profile windows build
docker compose --profile nemo build parakeet
docker compose --profile ocr build visual
```

Redo **titles only**, or **dialogue only**:

```powershell
docker compose --profile ocr run --rm visual
docker compose --profile nemo run --rm -e SUBTITLEGEN_OVERWRITE=1 parakeet
```

Outputs land next to the videos: `episode.srt` (Parakeet) and `episode.ass`
(dialogue + `OnScreen`).

Force the One Piece glossary on the title pass:

```powershell
docker compose --profile ocr run --rm visual generate /data/videos --profile one-piece --arc Dressrosa
```

### 3. Whisper instead of Parakeet (one container)

If you want Whisper `quality` and titles in a **single** run (no Parakeet):

```powershell
docker compose --profile ocr run --rm visual
```

That uses faster-whisper large-v3 in the OCR image, then Paddle/NLLB.

Compose already passes `generate /data/videos`. Extra flags:

```powershell
docker compose --profile ocr run --rm visual generate /data/videos --preset fast --no-visual-text
docker compose --profile ocr run --rm visual generate /data/videos --profile one-piece --arc Dressrosa
```

### 4. Optional native Windows (no Docker)

Only if you have Python 3.11/3.12 and do **not** need Paddle titles:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
.\.venv\Scripts\subtitlegen.exe generate C:\Videos\episode.mp4 --preset fast --no-visual-text
```

CUDA WhisperX: `pip install -e ".[cuda]"`. OCR extras on Windows are
unsupported here; use the `visual` image.

---

## Commands

```text
subtitlegen generate PATH
    PATH = one video or a directory (recursive).

    --preset quality|fast|english-fast
    --backend auto|mlx|faster-whisper|whisperx|parakeet
    --config config.ini
    --cache-dir .subtitlegen
    --output-dir DIR          # default: beside each video
    --overwrite               # rebuild SRT (and then ASS)
    --profile one-piece       # else inferred from the path
    --profiles-dir DIR
    --arc Dressrosa --episode 03
    --auto-profile / --no-auto-profile
    --local-correction / --no-local-correction
    --visual-text / --no-visual-text
    --visual-probe-seconds 4
    --visual-refine-seconds 12
    --visual-fps 1.5
    --visual-min-japanese-characters 5
    --detector-model comic-dbnet.onnx
    --verbose

subtitlegen validate SUBTITLE.srt
subtitlegen benchmark VIDEO [--preset quality] [--backend auto]
```

These `generate` flags also read env vars (Compose forwards the ones in `.env`):
`--preset`, `--backend`, `--config`, `--cache-dir`, `--output-dir`, `--profile`,
`--arc`, `--overwrite`, `--visual-fps`, `--visual-probe-seconds`,
`--visual-refine-seconds`. `SUBTITLEGEN_ENRICH_GLOSSARY` is env-only.
Run `subtitlegen generate --help` for the rest.

### Examples

```bash
# One episode, best local speech model, titles on
subtitlegen generate "samples/[Muhn Pace] Dressrosa 01.mp4" --preset quality --overwrite

# A season folder; skip titles
subtitlegen generate ./videos --preset fast --no-visual-text

# Force One Piece glossary
subtitlegen generate ./videos --profile one-piece --arc Dressrosa --preset quality
```

```powershell
# Windows: Parakeet SRT, then OCR ASS (rebuild after src/ changes)
docker compose --profile windows build
docker compose --profile windows up
docker compose --profile windows down

# Windows: dialogue only
docker compose --profile nemo build parakeet
docker compose --profile nemo run --rm parakeet

# Windows: Whisper quality + titles in one container
docker compose --profile ocr run --rm visual
```

---

## Config files

**`config.ini`** (native and mounted into Docker as `/app/config.ini`):

- `[TRANSCRIPTION]` — `device`, `model_name`, `language` (`en`), `compute_type`,
  `beam_size`, `whisperx_batch_size`. If VRAM dies on 8 GB, lower
  `whisperx_batch_size` or use `--preset fast`.
- `[VAD]` — silence / speech padding.
- `[CUES]` — max cue duration (6 s), max characters (84), gap flatten.
- `[FILES]` — video extensions.

CLI `--preset` **overrides** `model_name` from this file.

**`.env`** — Docker host paths and `SUBTITLEGEN_*` only. Not used by native
`subtitlegen` unless you export those variables yourself.

**`profiles/*.yaml`** — shipped glossaries (`one-piece`, `avatar`) plus
`visual_translations` (exact / fuzzy Japanese → English title lines).

---

## Cache and overwrite

| Artifact | Reused when |
|---|---|
| `.srt` | File exists and parses as valid SRT |
| ASR job under `--cache-dir/jobs` | Same backend + model + decode key |
| Visual job | Same detector, OCR, NLLB, profile, and `title-scan-v7` key |

`--overwrite` rebuilds the SRT (and then the ASS). Changing the visual cache
key (detector, glossary, scan version) rebuilds titles even if the SRT is
kept.

---

## On-screen titles (current behavior)

Default **on**. Probe every 4 s and on cuts for title-script; refine at 1 s
with a fresh detect (boxes are not frozen). Horizontal cards: furigana mask +
Paddle rec. Manga OCR only for tall vertical crops. Translation: profile
`visual_translations` first, then local NLLB-200 600M.

NLLB’s license is **non-commercial** for many uses. See
[docs/10-model-licenses.md](docs/10-model-licenses.md).

More detail: [docs/08-visual-text.md](docs/08-visual-text.md).

---

## Tests and acceptance

```bash
python -m pip install -e '.[mac,ocr,dev]'   # or .[dev] in the test image
ruff check .
mypy src
pytest -q
```

Windows RTX batch (builds images, CUDA check, sample episodes):

```powershell
pwsh ./scripts/windows-rtx-acceptance.ps1 `
  -VideoPath "D:\Anime" `
  -AvatarFile "Avatar Episode.mp4" `
  -DressrosaFile "[Muhn Pace] Dressrosa 03.mp4"
```

---

## More docs

| Doc | Topic |
|---|---|
| [docs/07-asr-backends.md](docs/07-asr-backends.md) | Presets, WhisperX, Parakeet |
| [docs/08-visual-text.md](docs/08-visual-text.md) | OCR / NLLB / ASS |
| [docs/09-platform-setup.md](docs/09-platform-setup.md) | Mac + Windows RTX setup |
| [docs/06-series-profiles.md](docs/06-series-profiles.md) | Glossaries |
| [docs/10-model-licenses.md](docs/10-model-licenses.md) | Model licenses |
