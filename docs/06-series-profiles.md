# 5. Series Profiles

Profiles are local, versioned YAML files. You do not write them by hand for a
normal run. `subtitlegen generate VIDEO_OR_DIRECTORY` infers the series from the
file or folder name, then resolves a glossary in this order:

1. A shipped profile whose id or title matches (`profiles/one-piece.yaml`, `profiles/avatar.yaml`).
2. A previously built profile in `.subtitlegen/profiles/`.
3. A new profile from Wikipedia (character lists and extracts) plus a fast web
   search if Wikipedia is thin.
4. A title-only profile, then a local transcript miner that adds repeated proper
   nouns after the first episode and writes them back to the cache.

`--profile` remains an override. `--no-auto-profile` turns the lookup off.
Local correction is on whenever a profile exists. Japanese visual text starts
automatically when the resolved profile looks like anime or already has visual
translations; use `--no-visual-text` to skip that pass.

Each term has a canonical spelling, optional aliases, a category, optional
arc/episode scope, and an alias-normalization safety flag. Auto-built single
English words such as `Law` stay prompt-only so ordinary dialogue is not rewritten.

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
subtitlegen generate VIDEO_OR_DIRECTORY
subtitlegen generate VIDEO --profile one-piece --arc Dressrosa
subtitlegen generate VIDEO --no-auto-profile --no-local-correction
```

The selected scope produces a bounded ASR prompt and hotword list. MLX keeps the initial profile prompt in its decoding history across audio windows. A separate deterministic pass replaces safe complete aliases only; ambiguous aliases such as “hockey” are prompt-only to avoid changing ordinary English. Conservative local correction runs on low-confidence cues. Media files stay on the machine; only the inferred series title is sent to Wikipedia or web search.

See [requirements](01-requirements.md), [upgrade choices](03-upgrade-choices.md), and [Phase 2 evidence](benchmarks/phase-2-context.md).
