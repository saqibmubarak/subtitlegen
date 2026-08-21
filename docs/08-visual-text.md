# Japanese visual text

Japanese on-screen translation starts automatically. Use `--no-visual-text` to
skip it, or `--visual-text` to force it.

```bash
subtitlegen generate episode.mp4
subtitlegen generate episode.mp4 --no-visual-text
```

OCR does **not** run Manga OCR on every frame. A mobile Japanese recognizer first
probes on scene changes and every `--visual-probe-seconds` (default 4). If any
Japanese character is found, the pipeline densifies `--visual-refine-seconds`
(default 12) before and after that hit at `--visual-fps` (1–2). A 30-second-only
grid would miss typical 3–8 second location cards, so scene cuts are part of the
coarse scan. Probe and dense sampling read every decoded frame rather than
keyframe-only (`NONREF`) skips, so a held card is not dropped between references.
The coarse scanner reads up to 16 text boxes per probe, not only the largest
ones, so an English `Scene-4` label does not hide a smaller Japanese line.
Vertical (`tate-gaki`) lines are detected as tall boxes. The coarse Paddle
recognizer rotates those crops 90° clockwise so a horizontal CRNN can read
them; Manga OCR then sees the original tall crop, which it already handles.
Nearby vertical and horizontal boxes on the same card are clustered into one
event: vertical columns right-to-left, then horizontal lines left-to-right.
Docker reads these from `SUBTITLEGEN_VISUAL_PROBE_SECONDS` and
`SUBTITLEGEN_VISUAL_REFINE_SECONDS`.

Inside a refine window the default PaddleOCR detector is used before Manga OCR.
Pass `--detector-model comic-dbnet.onnx` to try an anime/comic DBNet model with
PaddleOCR as its fallback.

Before running a model detector, the pipeline compares low-resolution adjacent
frames and proposes at most two changing regions. Proposals persist briefly so
static title text is observed more than once. Regions are padded, normalized,
and batched for Paddle/DBNet; Manga OCR only sees detector-confirmed crops.
Stable detector-confirmed regions are compared perceptually and reused without
rerunning either model.

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
