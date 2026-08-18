# ASR backends and presets

All ASR adapters return the same immutable `Transcription` and `Word` values. Cue
construction, profiles, correction, caching, and writers therefore do not depend
on a model framework.

## Presets

- `fast`: MLX large-v3-turbo on Apple Silicon, faster-whisper fp16 on CUDA, or
  faster-whisper int8 on CPU.
- `quality`: MLX large-v3 on Apple Silicon, WhisperX large-v3 forced alignment
  on CUDA, or faster-whisper large-v3 on CPU.
- `english-fast`: Parakeet TDT 0.6B v3 on CUDA and the fast platform choice
  elsewhere. The final 8 GB CUDA choice remains subject to Phase 5 measurement.

Use a preset without also specifying a backend:

```bash
subtitlegen generate /path/to/videos --preset quality
```

Use `--backend faster-whisper`, `mlx`, `whisperx`, or `parakeet` for an explicit
adapter. Unsupported hardware and missing optional dependencies fail with an
actionable error; selection never silently falls back.

WhisperX and Parakeet run in isolated Docker Compose profiles:

```bash
docker compose --profile whisperx run --rm whisperx
docker compose --profile nemo run --rm parakeet
```

WhisperX consumes profile prompts and hotwords before forced alignment. The
current Parakeet adapter is English-only and rejects series context instead of
silently ignoring it.

If WhisperX exhausts an 8 GB GPU, lower `whisperx_batch_size` in the
`[TRANSCRIPTION]` section of `config.ini` or select `fast`.

## Extension point

Implement `AsrBackend`, including capabilities, normalized transcription, and
resource release. Register the constructor in `BackendFactory`, then apply the
shared backend contract tests to the adapter.
