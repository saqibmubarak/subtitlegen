# 1. Requirements

## Functional

1. Generate synchronized dialogue subtitles from English-dubbed video.
2. Preserve series terminology and proper nouns through reusable profiles that are created automatically from the input path.
3. Detect Japanese on-screen text, translate it, and distinguish it from dialogue.
4. Export **ASS** as the primary format and dialogue-only **SRT** as an option.
5. Process a file or recursively process a directory while skipping valid outputs.

## Quality

- Dialogue cues should usually last 1–6 seconds and never bridge long silent gaps.
- Timing must follow spoken words, not ASR segment boundaries.
- On-screen translations must retain the visual text's appearance interval.
- Terminology must be consistent across ASR, OCR translation, and correction.
- Every automated stage must retain intermediate results for review and retries.

## Deployment

| Device | Required path |
|---|---|
| RTX 3070 Ti 8 GB | Windows Docker with NVIDIA GPU access |
| MacBook Pro M4 | Native Apple Silicon execution |
| CPU-only fallback | Smaller/quantized ASR with reduced speed |

The repository must use configurable host paths, persistent model caches, unbuffered structured logs, and one GPU worker by default. Defaults must not contain machine-specific paths or platform-only workarounds.

## Performance

- Batch within one loaded ASR model; do not load multiple large model copies on the 8 GB GPU.
- Run ASR, OCR, and LLM correction sequentially on one GPU.
- Allow independent files or stages to run across the RTX machine and Mac.
- Detect text on sparse/scene-change frames before running OCR.

## Acceptance

The completed product must:

1. Run from a clean Windows checkout after Docker/NVIDIA prerequisites are installed.
2. Run natively from a clean Mac checkout.
3. Produce styled ASS and optional SRT beside each video.
4. Pass the timing checks in [the sync proposal](02-sync-fix.md).
5. Improve named entities with a selected series profile.
6. Produce distinct translated on-screen events on representative One Piece samples.

Next: [Sync fix](02-sync-fix.md).
