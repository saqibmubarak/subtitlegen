# 4. Delivery Plan

## Final architecture

```text
series profile ───────────────┬─────────────────────┐
                              ▼                     ▼
video → audio → ASR → word cues → subtitle rules   │
  └→ sparse frames → text detect → OCR → translate │
                              └──────────────┬──────┘
                                             ▼
                                ASS merge + SRT export
                                             │
                                  optional gated correction
```

ASS contains `Dialogue` and `OnScreen` styles. Intermediate audio, word timestamps, OCR detections, recognized Japanese, translations, and metrics are cached per job.

## Phases

| Phase | Deliverable | Dependency | Completion test |
|---|---|---|---|
| 0 | Word-timestamp sync fix | Current faster-whisper | Avatar timing targets pass |
| 1 | Windows Docker and Mac-native baseline | Phase 0 | Clean-start smoke tests pass |
| 2 | Series profiles and context injection | Phase 1 | Proper-noun benchmark improves |
| 3 | WhisperX/Parakeet evaluation and selectable backends | Phase 2 | Device benchmark selects presets |
| 4 | Japanese visual-text pipeline and ASS merge | Phase 2 | One Piece sample passes review |
| 5 | Packaging, tests, documentation | 0–4 | Out-of-box acceptance passes |

## Parallelization

1. Load one ASR model per device.
2. Batch chunks within the model instead of starting multiple GPU processes.
3. Queue files sequentially on the 8 GB GPU.
4. Run OCR only on detector-positive sparse frames.
5. Run correction only on low-confidence or glossary-conflicting text.
6. Optionally let the RTX machine process ASR while the Mac processes OCR for another episode.

This maximizes throughput without duplicating model memory. Actual batch sizes are selected by measured peak VRAM, not fixed globally.

## Platform packaging

### Windows

- CUDA Docker image with pinned faster-whisper/CTranslate2-compatible libraries.
- Docker Compose GPU reservation enabled.
- `${VIDEO_HOST_PATH:-./videos}` mount; no checked-in absolute paths.
- Persistent model/output cache and `PYTHONUNBUFFERED=1`.
- Default: `language=en`, `float16`, one worker for English dubs.
- Optional legacy CUDA image only if host-driver testing requires it.

### Mac

- Native Apple Silicon environment; Docker is not the GPU path.
- MLX Whisper backend or tested CPU fallback.
- Same CLI, profiles, cache schema, output schema, and validation.

Target CLI:

```bash
docker compose run --rm subtitler generate /data/videos --profile one-piece --preset quality
subtitlegen generate ./videos --profile avatar --preset quality
```

## Presets

| Preset | Backend | Intended use |
|---|---|---|
| `fast` | large-v3-turbo word cues | Drafts |
| `quality` | large-v3 + optional WhisperX | General final subtitles |
| `english-fast` | Parakeet if benchmark-approved | English-dub batches |
| `mac-quality` | MLX large-v3 | Native M4 processing |

## Final acceptance

1. Windows and Mac setup guides reproduce the same output schema.
2. Phase 0 timing metrics pass.
3. Runtime and peak memory are recorded for each preset.
4. Profiles improve selected terms without increasing general errors materially.
5. One Piece location/name cards become distinct, correctly timed ASS events.
6. Failed stages resume from cached intermediates.
7. No platform-specific paths or temporary backend workarounds remain in defaults.

Sources and rationale: [References](05-references.md). Return to [contents](README.md).
