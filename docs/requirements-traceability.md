# Requirements Traceability

| Requirement | Implementation | Evidence |
|---|---|---|
| Word-aligned dialogue | `FasterWhisperBackend`, `CueBuilder` | `test_faster_whisper.py`, `test_cues.py`, [Mac evidence](benchmarks/phase-0-mac.md) |
| Bounded, non-overlapping cues | `CueRules`, `CueBuilder` | property and boundary tests |
| SRT output | `SrtWriter` | writer golden test |
| One GPU model by default | typed settings, backend instance lifecycle | settings/backend tests |
| Recursive input and skip support | compatibility CLI; portable CLI in Phase 1 | Phase 1 CLI tests |
| Windows Docker and Mac native | Phase 1 runtime adapters | Phase 1 smoke tests |
| Series terminology | Phase 2 profiles | Phase 2 proper-noun benchmark |
| Interchangeable ASR | Phase 3 backend contracts | Phase 3 adapter suite |
| Japanese on-screen translation | Phase 4 visual pipeline | Dressrosa annotations and ASS tests |
| Resumable intermediates | Phase 1 job store | interruption/resume tests |

This table is updated before each phase commit. See [requirements](01-requirements.md).
