# Subtitlegen Proposal

A phased plan for accurate dialogue subtitles, series-aware terminology, and translated Japanese on-screen text.

## Read in order

1. [Requirements](01-requirements.md)
2. [Sync fix](02-sync-fix.md)
3. [Upgrade choices](03-upgrade-choices.md)
4. [Delivery plan and final architecture](04-delivery-plan.md)
5. [Series profiles](06-series-profiles.md)
6. [ASR backends and presets](07-asr-backends.md)
7. [Japanese visual text](08-visual-text.md)
8. [References](05-references.md)
9. [Requirements traceability](requirements-traceability.md)

## Target devices

- **NVIDIA RTX 3070 Ti (8 GB):** primary Windows Docker batch processor.
- **MacBook Pro M4:** native portable processing, OCR review, and glossary correction.

## Recommended direction

Fix timing with faster-whisper first. Then add glossary profiles, validate WhisperX versus Parakeet on representative episodes, and finally add Japanese visual-text extraction with ASS output.
