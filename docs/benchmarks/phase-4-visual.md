# Phase 4 visual-text evidence

Date: 2026-08-18  
Device: MacBook Pro, Apple M4 Pro, 48 GB unified memory  
Media: five local `[Muhn Pace] Dressrosa` videos (media is not committed)

## Reproducibility

The lightweight annotations are
`tests/fixtures/dressrosa_visual_annotations.yaml` (SHA-256
`7a54bc2a4d17563c724d91b512cb975f8a6b1e28704c2317534d08347b7f6a03`).
They reference seven manually reviewed high-value cards and 85 seconds of
negative windows across all five videos. Coordinates use the 1920x1080 source
space. Decorative effects and background signage are explicitly deferred.

Model revisions:

- Manga OCR `kha-white/manga-ocr-base`:
  `aa6573bd10b0d446cbf622e29c3e084914df9741`
- NLLB `facebook/nllb-200-distilled-600M`:
  `f8d333a098d19b4fd9a8b18f94170487ad3f821d`
- PaddleOCR 3.7.0 `PP-OCRv5_mobile_det`

Run the acceptance suite with cached models and local media:

```bash
SUBTITLEGEN_RUN_MODEL_TESTS=1 \
HF_HOME="$PWD/.subtitlegen/hf" \
PADDLE_PDX_CACHE_HOME="$PWD/.subtitlegen/paddle" \
.venv/bin/pytest -q tests/model/test_dressrosa_visual.py -vv
```

## Results

The fixture-driven model test passed both acceptance cases.

| Category | Cards | Detection recall | Mean accepted-crop CER |
|---|---:|---:|---:|
| Location cards | 2 | 100% | 0.0% |
| Character cards | 4 | 100% | 5.6% |
| Scene cards | 1 | 100% | 0.0% |
| Decorative text (deferred) | 0 | not scored | not scored |

Canonical profile translation consistency was 7/7 (100%). The tracked
Kin'emon scene card was emitted at 1319.000–1322.333 against the manually
annotated 1318.500–1322.250 interval, for appearance-interval IoU 0.848.

The 85 seconds of reviewed negative windows emitted zero persistent events
(0.0 false positives/minute) after applying the high-value-card size and
vertical-position policy. Before that policy, Manga OCR converted arena texture
and motion lines into 38 persistent false events; this failure is retained here
to make the precision trade-off explicit.

The cold six-second scene-card run took 21.69 seconds, including Paddle,
Manga OCR, and NLLB model loading. The 85-second negative-window run took
42.27 seconds with Paddle and Manga OCR cold loading and no translation model
load. These window timings are diagnostic and are not extrapolated to a full
episode because scene density and OCR candidate count dominate runtime.

The complete two-test model gate took 61.38 seconds wall time under
`/usr/bin/time -l`, with maximum resident set size 4.83 GB. This is unified
process memory on macOS, not CUDA VRAM.

## Interpretation

The initial targets are met for the prioritized card classes: detection recall
is at least 90%, accepted-crop CER is at most 10% per reported class, timing IoU
is at least 0.8, and canonical glossary consistency is at least 95%. The result
does not claim quality for decorative effects, arbitrary signs, wanted posters,
or maps outside the annotated set. Expanding those classes requires a larger
annotation set and a detector tuned for anime typography without relaxing the
current false-positive guard.
