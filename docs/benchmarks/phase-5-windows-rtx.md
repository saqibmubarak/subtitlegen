# Phase 5 Windows RTX acceptance

Status: **not executed on the physical RTX 3070 Ti**

The current development host is macOS/Apple Silicon and cannot provide Windows,
WSL2, NVIDIA driver, CUDA visibility, or 8 GB VRAM measurements. Compose syntax
and local non-CUDA gates can be checked here, but that is not equivalent to the
required target-device acceptance. Do not mark Phase 5 complete or tune presets
until this report is replaced with captured target-host evidence.

## Target host inventory

- Date:
- Windows version:
- Docker Desktop version:
- NVIDIA driver:
- GPU reported by `nvidia-smi`:
- Clean checkout commit:

## Required command

```powershell
pwsh ./scripts/windows-rtx-acceptance.ps1 `
  -VideoPath "D:\Anime With Spaces" `
  -AvatarFile "Avatar Episode.mp4" `
  -DressrosaFile "[Muhn Pace] Dressrosa 03.mp4"
```

## Results to capture

| Gate | Verification method | Result/evidence |
|---|---|---|
| CUDA visible in PyTorch container | script assertion | pending |
| Baseline, WhisperX, NeMo, visual images build cleanly | script build | pending |
| Unit/integration tests pass in CUDA, NeMo, OCR images | script tests | pending |
| fp16 faster-whisper and one GPU worker | manual log plus `nvidia-smi` review | pending |
| Host paths with spaces | script bind mount | pending |
| Recursive duplicate stems | containerized CLI unit test | pending |
| First run, resume, skip, corrupt-artifact recovery | script reruns and JSON parse | pending |
| Explicit CPU/int8 fallback | script assertion by successful run | pending |
| Avatar runtime, RTFx, peak VRAM, cue metrics, glossary | script output plus manual capture | pending |
| Dressrosa runtime, peak VRAM, OCR/CER/timing, ASS playback | script output plus manual review | pending |
| Mac/Windows schema comparison | manual normalized JSON/hash comparison | pending |

Attach or paste command output and record output hashes. Any OOM must be resolved
from measured 8 GB behavior, not by silently changing backend or quality.
