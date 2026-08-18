# Requirements Traceability

| Requirement | Implementation | Evidence |
|---|---|---|
| Word-aligned dialogue | `FasterWhisperBackend`, `CueBuilder` | `test_faster_whisper.py`, `test_cues.py`, [Mac evidence](benchmarks/phase-0-mac.md) |
| Bounded, non-overlapping cues | `CueRules`, `CueBuilder` | property and boundary tests |
| SRT output | `SrtWriter` | writer golden test |
| One GPU model by default | typed settings, backend instance lifecycle | settings/backend tests |
| Recursive input and skip support | Typer CLI and `RuntimeService` | CLI/service tests |
| Windows Docker and Mac native | Docker CUDA profile and MLX adapter | [Phase 1 Mac evidence](benchmarks/phase-1-mac.md), Compose validation |
| Series terminology | Versioned profiles, scoped context, safe normalization | [Phase 2 context evidence](benchmarks/phase-2-context.md), terminology fixture |
| Interchangeable ASR | Four normalized adapters and capability-aware presets | shared adapter contracts, [Phase 3 evidence](benchmarks/phase-3-backends.md) |
| Japanese on-screen translation | Detection, Manga OCR, local NLLB/profile translation, tracking, and ASS merge | [Phase 4 evidence](benchmarks/phase-4-visual.md), Dressrosa model tests, ASS golden/round-trip tests |
| Resumable intermediates | `PortableJobStore`, `StageExecutor` | job/executor/service tests |

This table is updated before each phase commit. See [requirements](01-requirements.md).
