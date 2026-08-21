# Japanese visual text

Japanese on-screen translation starts automatically. Use `--no-visual-text` to
skip it, or `--visual-text` to force it.

```bash
subtitlegen generate episode.mp4
subtitlegen generate episode.mp4 --no-visual-text
```

OCR does **not** run Manga OCR on every frame. A mobile Japanese recognizer first
probes on scene changes and every `--visual-probe-seconds` (default 4). A probe
is a hit only when a crop has **title script** (two or more kanji, or three or
more katakana). One-character noise and hiragana filler such as `そういえば` do
not open a refine window. Hits keep the **exact detector boxes**, not a padded
half-frame.

Around each hit the pipeline walks `--visual-refine-seconds` (default 12) at a
one-second crop-hash interval. Manga OCR and NLLB run only when that crop
changes. If the title is still the same, the previous translation is reused.
A 30-second-only grid would miss typical 3–8 second location cards, so scene
cuts are part of the coarse scan. Probe sampling reads every decoded frame
rather than keyframe-only (`NONREF`) skips, so a held card is not dropped
between references. The coarse scanner reads up to 16 text boxes per probe,
not only the largest ones, so an English `Scene-4` label does not hide a
smaller Japanese line. Vertical (`tate-gaki`) lines are detected as tall
boxes. The coarse Paddle recognizer rotates those crops 90° clockwise so a
horizontal CRNN can read them; Manga OCR then sees the original tall crop.
Nearby vertical and horizontal boxes on the same card are clustered into one
event: vertical columns right-to-left, then horizontal lines left-to-right.
Docker reads these from `SUBTITLEGEN_VISUAL_PROBE_SECONDS` and
`SUBTITLEGEN_VISUAL_REFINE_SECONDS`.

Pass `--detector-model comic-dbnet.onnx` to try an anime/comic DBNet model with
PaddleOCR as its fallback. Detector boxes use a modest `unclip_ratio` so the
OCR crop stays on the glyphs. If a refine frame has no saved boxes, a small
motion proposal (about 8% padding, not 50% of the frame) is the fallback.

The precision policy then keeps large cards, requires at least five Japanese
characters, and rejects Manga OCR filler that is only conversational hiragana.
A keep candidate needs **two or more kanji** (name/place characters such as
`一人` / `錦`) **or three or more katakana** (`ドレスローザ`). That drops
`そういえば` and `人のところで` while keeping real location cards. Upper-frame
cards are no longer dropped by default. Change the Japanese-character floor with
`--visual-min-japanese-characters`.

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

For Docker, copy `.env.example` to `.env` and run:

```bash
docker compose --profile ocr run --rm visual
```

See the [Phase 4 benchmark](benchmarks/phase-4-visual.md) for the annotation protocol,
quality measurements, and current scope.
