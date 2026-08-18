# 5. Series Profiles

Profiles are local, versioned YAML files in `profiles/`. Each term has a canonical spelling, optional aliases, a category, optional arc/episode scope, and an alias-normalization safety flag.

```yaml
schema_version: 1
profile_id: one-piece
title: One Piece
language: en
terms:
  - canonical: Doflamingo
    aliases: [Don Quixote do Flamingo]
    category: character
    arcs: [Dressrosa]
```

Use a profile with:

```bash
subtitlegen generate VIDEO --profile one-piece --arc Dressrosa
subtitlegen generate VIDEO --profile one-piece --local-correction
```

The selected scope produces a bounded ASR prompt and hotword list. MLX keeps the initial profile prompt in its decoding history across audio windows. A separate deterministic pass replaces safe complete aliases only; ambiguous aliases such as “hockey” are prompt-only to avoid changing ordinary English. Optional conservative local correction is called only for low-confidence cues. No profile data or media leaves the machine.

See [requirements](01-requirements.md), [upgrade choices](03-upgrade-choices.md), and [Phase 2 evidence](benchmarks/phase-2-context.md).
