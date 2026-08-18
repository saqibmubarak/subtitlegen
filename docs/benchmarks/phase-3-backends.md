# Phase 3 backend evidence

Device: MacBook Pro M4 Pro, 48 GB unified memory.

Samples:

- `avatar-114-126.wav`, SHA-256
  `6d585d0808a1dff23b29e30b028009e5715ea55287b3017571326fa65da22ff5`;
  committed phrase text/onsets are in `tests/fixtures/avatar_asr_annotations.yaml`.
- Local full Avatar video, 5,931.05 seconds. Dialogue ends at about 01:27:59;
  later model output is counted as a false positive rather than completeness.

Pinned model revisions:

- MLX large-v3-turbo: `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`.
- MLX large-v3: `49e6aa286ad60c14352c404340ded53710378a11`.
- faster-whisper large-v3-turbo:
  `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`.

## Short sample

- MLX turbo: 3.37 seconds, RTF 0.280, 2.10 GB maximum RSS, WER 0,
  terminology recall 1.0, phrase-onset MAE 0.293 seconds.
- MLX large-v3 warm: 4.39 seconds, RTF 0.365, 3.89 GB maximum RSS, WER 0,
  terminology recall 1.0, phrase-onset MAE 0.430 seconds.
- faster-whisper CPU int8: 6.87 seconds, RTF 0.572, 2.49 GB maximum RSS,
  WER 0, terminology recall 1.0, phrase-onset MAE 0.
- Every output had three cues, zero overlaps, and zero cues over eight seconds.
- The first uncached MLX large-v3 invocation took 78.99 seconds including model
  download; its warm invocation above separates inference from download time.

Output SHA-256 values:

- MLX turbo:
  `7e625cdd371ccaa79ada8255ce0471f6eaabfac9882169b926307fa05bb876d5`.
- MLX large-v3:
  `0b8a797a3973e1a1223d11e89ca3656d87a5508344908a85ccc3fc1ddf572038`.
- faster-whisper:
  `f8a12cfaa025d7d6635c008f82db2d664a2892c8a294feee64a8ed7c6a26e78a`.

## Full sample

- MLX large-v3: 492.34 seconds, RTF 0.083, 5.13 GB maximum RSS, 920 cues,
  2.27-second median, zero overlaps, and 34 false cues after dialogue ended.
- faster-whisper CPU int8: 397.24 seconds, RTF 0.067, 10.55 GB maximum RSS,
  656 cues, 2.33-second median, zero overlaps, and no cue after dialogue ended.
- The earlier MLX turbo full run took 350.47 seconds (RTF 0.059). Its retained
  output has 50 false cues after dialogue ended, so speed does not hide the
  long-silence hallucination difference.

Full output SHA-256 values:

- MLX large-v3:
  `cf7af41f559857c8d60e133bf7c86dcbf6b3e7d145f1aef25f20ed8940407871`.
- faster-whisper:
  `38fa3efee180c5bfad07747777a3a4dc79bb5d8b66f46f8e14cd99bbe78e3861`.

Commands used:

```bash
HF_HOME="$PWD/.subtitlegen/hf" subtitlegen benchmark avatar-114-126.wav --preset fast
HF_HOME="$PWD/.subtitlegen/hf" subtitlegen benchmark avatar-114-126.wav --preset quality
subtitlegen benchmark avatar-114-126.wav --backend faster-whisper
HF_HOME="$PWD/.subtitlegen/hf" subtitlegen benchmark LegendOfAang.mp4 --preset quality
subtitlegen benchmark LegendOfAang.mp4 --backend faster-whisper
```

WhisperX and Parakeet are CUDA-only and were contract-tested with mocked model
APIs on Mac. Their runtime, VRAM, alignment, and quality comparison remains an
explicit Phase 5 RTX acceptance item; presets fail clearly on this Mac.
