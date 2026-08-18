# Phase 5 Mac hardening

Run date: 2026-08-18 on Apple Silicon, macOS 25.3.0.

## Local gates

- Clean Python 3.11 temporary environment: package plus `dev` extra installed,
  115 non-model tests passed, the optional OpenCV test skipped as expected, and
  `subtitlegen --help` succeeded.
- Development Python 3.12 environment: 116 tests passed, two opt-in model
  tests skipped, and 88.65% branch-aware coverage passed the 85% gate.
- Ruff: all files passed.
- Mypy strict mode: 54 source files passed.
- `pip check`: no broken requirements.
- Compose configuration: valid under Docker 29.6.2.
- Dressrosa opt-in model gate: two tests passed in 53.97 seconds with 4.15 GB
  maximum resident memory and the final NumPy 2.2.6, Transformers 4.57.6,
  PaddleOCR 3.7.0, and Manga OCR 0.1.16 pins.

## Linux container gates from Mac

Docker Desktop used amd64 emulation; this validates clean image construction
and Linux package resolution, not NVIDIA behavior.

- Baseline runtime image built and `subtitlegen --help` succeeded.
- CUDA/WhisperX test image built; final mounted source: 115 passed, one
  OCR-extra-dependent skip.
- CUDA/OCR test image built; final mounted source: 116 passed.
- NeMo test image built; final mounted source: 115 passed, one
  OCR-extra-dependent skip.
- Production visual image built; imports for WhisperX, PaddleOCR, Manga OCR,
  Transformers 4.57.6, and packaged `one-piece` profile succeeded.

The builds exposed and fixed two real cross-extra conflicts: WhisperX requires
NumPy 2.1 or newer, and WhisperX's Hugging Face ceiling is incompatible with
Transformers 5. The unified pins are NumPy 2.2.6 and Transformers 4.57.6.

## Remaining target evidence

Windows/WSL2, CUDA visibility, fp16 execution, peak RTX VRAM, and full
cross-device output comparison require the physical target and remain pending
in [the Windows report](phase-5-windows-rtx.md).
