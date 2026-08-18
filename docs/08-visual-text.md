# Japanese visual text

`--visual-text` adds translated Japanese on-screen text to the generated
dialogue subtitles and writes an ASS file beside the dialogue-only SRT.

```bash
subtitlegen generate episode.mp4 \
  --profile one-piece \
  --arc Dressrosa \
  --visual-text \
  --visual-fps 1.5
```

The sampling rate accepts values from 1 through 2 fps. Scene changes are sampled
in addition to that cadence. The default PaddleOCR detector is used before
Manga OCR; pass `--detector-model comic-dbnet.onnx` to try an anime/comic DBNet
model with PaddleOCR as its fallback.

The initial precision policy intentionally keeps large lower-frame cards and
rejects small or upper-frame detections. This targets location, character, and
scene cards while deferring decorative sound effects and background signage.
The policy is resolution-independent and can be changed through
`VisualTextPipeline` when embedding the package.

Translation is local. NLLB-200 distilled 600M is the fallback, while exact and
high-confidence fuzzy matches from a series profile provide canonical
translations for known cards. Models download on first use and use their normal
local caches afterward.

The combined ASS contains separate `Dialogue` and `OnScreen` styles. The SRT is
kept as a dialogue-only compatibility output. Visual events are cached in the
portable job store and are regenerated when detector, profile, or sampling
settings change.

Install the optional dependencies with:

```bash
python -m pip install -e '.[ocr]'
```

For Docker, use the `visual` service defined in `docker-compose.yml`. See the
[Phase 4 benchmark](benchmarks/phase-4-visual.md) for the annotation protocol,
quality measurements, and current scope.
