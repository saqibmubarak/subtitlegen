# Japanese visual text

Japanese on-screen translation starts automatically. Use `--no-visual-text` to
skip it, or `--visual-text` to force it.

```bash
subtitlegen generate episode.mp4
subtitlegen generate episode.mp4 --no-visual-text
```

OCR does **not** run Manga OCR on every frame. A mobile Japanese recognizer first
probes every `--visual-probe-seconds` (default 3). Scene-change frames still
update a 0.25 s signature so refine can find cuts, but they do not run OCR.
A probe is a hit when a crop has **title script** (two or more kanji, or
three or more katakana), or when a **tall vertical** column has any Japanese
(or rotates into Japanese). Horizontal hiragana filler such as `そういえば` still
does not open a refine window. Hits keep the **exact detector boxes**, not a
padded half-frame.

Around each hit the pipeline walks `--visual-refine-seconds` (default 12) at a
one-second interval. Refine frames always re-detect text instead of locking the
probe crop coordinates: a newspaper headline and a lower-third board in the
same window must not share one frozen box. OCR and NLLB still reuse a result
when that crop's perceptual hash is unchanged. Horizontal HUD cards go through
Paddle recognition first, after a morphological furigana mask; Manga OCR is
only a fallback for tall vertical crops. Probe sampling reads every decoded
frame rather than keyframe-only (`NONREF`) skips, so a held card is not dropped
between references. The coarse scanner reads up to 32 text boxes per probe and inspects tall
columns before large English HUD boxes. Vertical (`tate-gaki`) lines are
detected as tall boxes and use a lower area floor than wide cards (~0.15% of
the frame instead of 1%). The coarse Paddle recognizer always rotates those
crops 90° clockwise so a horizontal CRNN can read them. Nearby vertical and
horizontal boxes on the same card are clustered into one event: vertical
columns right-to-left, then horizontal lines left-to-right. If every box in a
cluster fails OCR, the union crop is tried once. Docker reads probe/refine
windows from `SUBTITLEGEN_VISUAL_PROBE_SECONDS` and
`SUBTITLEGEN_VISUAL_REFINE_SECONDS`.

Paddle detect/rec run in a separate process when `paddlepaddle-gpu` is
installed (`SUBTITLEGEN_PADDLE_DEVICE=auto` on the Windows `cuda,ocr` image).
The GPU 3.3.1 wheel is pulled from Paddle's CUDA 12.9 index, not PyPI. The
parent process keeps Manga OCR and NLLB on PyTorch/CUDA. Set
`SUBTITLEGEN_PADDLE_DEVICE=cpu` to force the in-process CPU wheel.

Pass `--detector-model comic-dbnet.onnx` to try an anime/comic DBNet model with
PaddleOCR as its fallback. Detector boxes use a modest `unclip_ratio` so the
OCR crop stays on the glyphs. If a refine frame has no saved boxes, a small
motion proposal (about 8% padding, not 50% of the frame) is the fallback.

The precision policy then keeps large cards, requires at least three Japanese
characters, rejects Manga OCR filler that is only conversational hiragana, and
applies a final keep filter (title-script, glossary hit, drop dates / HUD
loanwords / dialogue-like English dumps, collapse duplicate cards).
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
kept as a dialogue-only compatibility output. A later run skips a video that
already has a valid `.ass` unless you pass `--overwrite`, and does not start
Paddle or NLLB when every file already has one. Visual events are also cached
in the portable job store and are regenerated when detector, profile, or
sampling settings change and the ASS is missing or overwrite is set.

Install the optional dependencies with:

```bash
python -m pip install -e '.[ocr]'
```

For Docker, copy `.env.example` to `.env` and run:

```bash
docker compose --profile ocr run --rm visual
```

Score a run against the Dressrosa fixture or Whole Cake expected names:

```bash
python scripts/score_visual_titles.py episode.titles.jsonl \
  --gold tests/fixtures/dressrosa_visual_annotations.yaml \
  --names tests/fixtures/whole_cake_island_01_expected_names.yaml
```

See the [Phase 4 benchmark](benchmarks/phase-4-visual.md) for the annotation protocol,
quality measurements, and current scope.
