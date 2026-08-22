# ASR backends and presets

All ASR adapters return the same immutable `Transcription` and `Word` values. Cue
construction, profiles, correction, caching, and writers therefore do not depend
on a model framework.

## Presets

- `fast`: MLX large-v3-turbo on Apple Silicon, faster-whisper turbo fp16 on CUDA,
  or faster-whisper turbo int8 on CPU. Faster, weaker names and grammar.
- `quality`: Whisper `large-v3`. On Apple Silicon that is MLX large-v3. On CUDA it
  is WhisperX large-v3 with forced alignment when `whisperx` is installed (the
  `whisperx` Compose image). The OCR/`subtitler` images do not include WhisperX, so
  quality there is faster-whisper large-v3. On CPU it is faster-whisper large-v3
  int8.
- `english-fast`: Parakeet TDT 0.6B v3 on CUDA and the fast platform choice
  elsewhere.

In Docker, set `SUBTITLEGEN_PRESET` in `.env`. Do not repeat flags on
`docker compose run`.

```bash
subtitlegen generate /path/to/videos --preset quality
docker compose --profile ocr run --rm visual
```

Use `--backend faster-whisper`, `mlx`, `whisperx`, or `parakeet` for an explicit
adapter. Missing optional packages fail with an actionable error. `quality` on
CUDA without WhisperX selects faster-whisper large-v3 instead of aborting, because
the OCR image is expected to run quality ASR without the WhisperX extra.

WhisperX and Parakeet run in isolated Docker Compose profiles. On Windows the
recommended path is one sequential job (`windows` runs Parakeet, then OCR
with `--reuse-srt` so titles never load WhisperX):

```bash
docker compose --profile windows up
docker compose --profile whisperx run --rm whisperx
docker compose --profile nemo run --rm parakeet
```

WhisperX consumes profile prompts and hotwords before forced alignment.
Those prefixes share Whisper's 448-token decoder window, so the adapter
caps them and retries without the glossary if a segment still overflows.
A second overflow falls back to faster-whisper for that file.
Parakeet is English-only and CUDA-only, so it cannot run on Mac (including
Docker Desktop). It runs decode, GPU, and SRT write as overlapping stages:
ffmpeg decodes the next files, the model stays loaded, and cue writing does
not block the next transcribe. All 20 s windows go to one batched NeMo call.
A VRAM fault halves the batch, then the window. Glossary correction still
runs after Parakeet transcription; the model itself does not consume
Whisper-style prompts.

If WhisperX exhausts an 8 GB GPU, lower `whisperx_batch_size` in the
`[TRANSCRIPTION]` section of `config.ini` or select `fast`.

## Extension point

Implement `AsrBackend`, including capabilities, normalized transcription, and
resource release. Register the constructor in `BackendFactory`, then apply the
shared backend contract tests to the adapter.
