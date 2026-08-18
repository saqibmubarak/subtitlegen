# Requirements traceability

| ID | Requirement | Implementation | Automated/manual evidence |
|---|---|---|---|
| F1 | Synchronized English dialogue | Normalized word timestamps and `CueBuilder` | cue properties, backend contracts, [Phase 0](benchmarks/phase-0-mac.md) |
| F2 | Reusable series terminology | Auto profile from path (shipped/cache/Wikipedia/search/transcript), scoped context, normalization, gated correction | profile suites, [Phase 2](benchmarks/phase-2-context.md) |
| F3 | Japanese visual translation | Temporal proposals, Paddle/DBNet, Manga OCR, local translator, tracker | visual suites, [Phase 4](benchmarks/phase-4-visual.md) |
| F4 | Styled ASS and dialogue SRT | `SubtitleMerger`, `AssWriter`, `SrtWriter` | ASS/SRT golden and round-trip tests |
| F5 | File/recursive batch and valid-output skip | Typer CLI, discovery, output metadata | paths-with-spaces, duplicate-stem, skip/overwrite tests |
| Q1 | Readable cues and silence boundaries | Typed cue/VAD rules | boundary/property tests and Avatar static analysis |
| Q2 | Visual appearance intervals | persistence tracker using configured sample interval | synthetic tracking and Dressrosa timing IoU |
| Q3 | Cross-stage terminology consistency | shared profile and visual translation map | terminology recall and 100% annotated canonical consistency |
| Q4 | Reviewable resumable intermediates | versioned JSON manifests and validated artifacts | resume, corruption, migration, cancellation, concurrency tests |
| D1 | RTX Windows Docker | CUDA, WhisperX, NeMo, OCR images and acceptance script | Compose validation; physical RTX run recorded in Phase 5 report |
| D2 | Native Apple Silicon | MLX adapter and PyAV media path | clean-install commands and Mac model evidence |
| D3 | CPU fallback | faster-whisper CPU/int8 configuration | factory tests and Windows acceptance script |
| D4 | Portable paths/caches/logging | environment mounts, POSIX artifact paths, JSON logs | path, cache, and formatter tests |
| P1 | One GPU owner/model lifecycle | `GpuResourceToken` and explicit backend/pipeline close | executor, service, and CLI cleanup tests |
| P2 | Sparse detection before OCR | 1–2 fps/scene sampler and temporal region proposer | sampler/proposer pipeline tests and Phase 4 timings |
| P3 | Interchangeable ASR | four adapters and capability-aware presets | shared contracts and [Phase 3](benchmarks/phase-3-backends.md) |
| A1 | Clean Windows checkout | documented prerequisites and PowerShell acceptance runner | [platform guide](09-platform-setup.md), Phase 5 Windows report |
| A2 | Clean Mac checkout | Python 3.11/3.12 extras and command smoke | [platform guide](09-platform-setup.md), clean-venv gate |
| A3 | Avatar and Dressrosa acceptance | dialogue benchmarks and visual annotations | Phase 0–4 evidence and final cross-device report |
| A4 | License visibility | pinned dependency extras and model manifest | [model/license manifest](10-model-licenses.md) |

The physical RTX rows are complete only when the commands and measurements in
the Phase 5 Windows report have been filled from the target machine. See the
source [requirements](01-requirements.md).
