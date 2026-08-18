# Phase 2 context evidence

Device/backend: MacBook Pro M4 Pro, MLX large-v3-turbo, local-only processing.

## Deterministic fixture

Four committed Avatar/One Piece terminology cases improved from 1/4 exact before normalization to 4/4 after normalization. The ordinary sentence “They played hockey under the sunny sky” remained unchanged because that ambiguous alias is prompt-only.

## Dressrosa dialogue clip

Sample: local Dressrosa episode 1, 00:10:00–00:12:00.

- Source SHA-256: `87e092284b7d6f75b9247e611e5bc59702b1e42a11e6372efd1202e2bd00c7dd`.
- Extracted 120-second WAV SHA-256: `88dfbac5f43b38f6ed0e2e98b8a99ac2cf07c04ec1e1ce8594a32cf8de2bda65`.
- MLX model revision: `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`.
- Final profiled SRT SHA-256: `96282f840630c51a003e5e5b80c1c29d0cdef7a14dd719300395bfd18fe49489`.
- Manual labels: [Dressrosa terminology annotations](../../tests/fixtures/dressrosa_terminology_annotations.yaml).

- Baseline: 7.48 seconds, RTF 0.062.
- Dressrosa profile with retained MLX prompt and local correction: 7.94 seconds, RTF 0.066.
- Annotated recurring-name spellings improved from 6/9 to 9/9.
- Corrected examples: “Don Quixote do Flamingo” → “Doflamingo”, “Raja” → “Roger”, and “Bucky” → “Buggy”.
- One ordinary grammatical phrase changed during prompted decoding; this is recorded rather than hidden. The three proper-name repairs outweighed that isolated regression on the reviewed clip.
- Both outputs had zero overlaps and zero cues over 8 seconds.
- Final profiled output: 23 cues, 3.44-second median, 5.04-second maximum.

Commands after extracting the source-relative interval with PyAV:

```bash
subtitlegen benchmark dressrosa-600-720.wav --backend mlx --cache-dir baseline
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
test "$(cat "$HF_CACHE/hub/models--mlx-community--whisper-large-v3-turbo/refs/main")" = \
  a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb
subtitlegen benchmark dressrosa-600-720.wav --backend mlx --profile one-piece \
  --arc Dressrosa --local-correction --cache-dir profiled
```

## Avatar clip

The 12-second Avatar clip retained “airbender” and “Avatar” with zero overlap or overlong cues. Profile processing took 3.05 seconds.
