# 3. Upgrade Choices

Public WER is only a shortlist signal. Subtitle quality also depends on timestamps, long-form behavior, domain vocabulary, noise, and hardware. Leaderboard datasets and averages are not always identical, and private evaluations can reorder models [R3][R4].

## ASR

| Backend | Timing | Speed | RTX 8 GB | M4 | Decision |
|---|---|---:|---:|---:|---|
| faster-whisper large-v3 | Word timestamps | High | Yes | CPU/alternative runtime | Phase 0 default |
| WhisperX + large-v3 | Forced word alignment | High, extra pass | Yes | CUDA-first | Quality candidate |
| Parakeet-TDT-0.6B-v3 | Native word/segment | Very high | Yes | Limited native support | English-dub candidate |
| large-v3-turbo | Word timestamps, weaker decoder | Very high | Yes | Yes through MLX | Fast preset |
| Canary-Qwen-2.5B | No shipped word timing | Medium | Tight | Poor | Exclude as primary |
| openai-whisper | Segment-oriented | Lower | Yes | Yes | No advantage here |

Parakeet provides native timestamps, punctuation, 25 European languages, and long-audio modes [R5]. WhisperX adds wav2vec2 forced alignment to faster-whisper [R2]. Canary-Qwen is English-only and was trained with audio up to 40 seconds; its model card does not provide the timing capability required here [R6].

**Decision process:** benchmark faster-whisper, WhisperX, and Parakeet on the same English-dub clips. Score word error, proper-noun error, timestamp error, cue readability, runtime, and peak memory. Do not select from leaderboard WER alone.

## Series context

| Method | Accuracy effect | Cost | Use |
|---|---:|---:|---|
| Whisper prompt/hotwords | Moderate | Negligible | First implementation |
| NeMo phrase boosting | Strong for listed terms | Small decode overhead | Parakeet path |
| Confidence-gated LLM correction | Strong but can over-edit | Additional pass | Optional final pass |

Store canonical names, aliases, places, and terms in `profiles/<series>.yaml`. Select only episode/arc-relevant terms for the short Whisper conditioning budget. Reuse the complete profile for OCR translation and correction. NeMo supports decoding-time phrase boosting without retraining [R7]; research shows context biasing and confidence-gated correction can improve rare terms while limiting unwanted edits [R8][R9].

## Japanese on-screen text

```text
scene-change or 1–2 fps samples
→ anime text detector
→ persistence/script/confidence filter
→ deduplicate crops
→ manga-ocr
→ glossary-aware translation
→ ASS OnScreen events
```

| Component | Preferred | Fallback | Reason |
|---|---|---|---|
| Detection | AnimeText-trained/DBNet comic detector | PaddleOCR | Anime typography differs from document text [R10] |
| Recognition | manga-ocr | PaddleOCR Japanese | Handles vertical text, furigana, and varied manga fonts [R11] |
| Translation | Local model + profile | API | Offline consistency versus higher external dependency |
| Output | ASS | SRT prefix | ASS supports position, color, and separate styles |

Prioritize location and character cards. Defer decorative sound effects until the high-value classes are reliable.

## Device allocation

- **RTX 3070 Ti:** batch ASR, WhisperX/Parakeet evaluation, heavy OCR.
- **M4:** MLX Whisper preset, OCR review, glossary extraction, optional local correction.
- **One 8 GB GPU:** unload between ASR, OCR, and correction stages.
- **Two machines:** process different files or stages concurrently through a shared job directory.

Next: [Delivery plan](04-delivery-plan.md). Sources: [References](05-references.md).
