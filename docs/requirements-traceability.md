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
| Interchangeable ASR | Phase 3 backend contracts | Phase 3 adapter suite |
| Japanese on-screen translation | Phase 4 visual pipeline | Dressrosa annotations and ASS tests |
| Resumable intermediates | `PortableJobStore`, `StageExecutor` | job/executor/service tests |

This table is updated before each phase commit. See [requirements](01-requirements.md).
