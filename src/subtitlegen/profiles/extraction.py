from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from subtitlegen.domain.models import Cue
from subtitlegen.profiles.composer import ProfileComposer, safe_normalization_flags
from subtitlegen.profiles.models import GlossaryEntry, SeriesProfile

_NAME = re.compile(r"\b([A-Z][A-Za-z]+(?:[ '][A-Z][A-Za-z]+){0,3})\b")
_STOP = frozenset(
    {
        "and",
        "but",
        "hey",
        "okay",
        "please",
        "thanks",
        "that",
        "the",
        "then",
        "there",
        "they",
        "this",
        "what",
        "when",
        "where",
        "yeah",
        "yes",
        "you",
    }
)


class LocalTranscriptExtractor:
    """Mine repeated proper nouns from finished cues without loading another model."""

    def __init__(self, composer: ProfileComposer | None = None, minimum_count: int = 2) -> None:
        if minimum_count < 1:
            raise ValueError("minimum count must be positive")
        self._composer = composer or ProfileComposer()
        self._minimum_count = minimum_count

    def extract(self, cues: Iterable[Cue]) -> tuple[GlossaryEntry, ...]:
        counts: Counter[str] = Counter()
        for cue in cues:
            for match in _NAME.finditer(cue.text):
                name = match.group(1).strip()
                if len(name) < 3 or name.casefold() in _STOP:
                    continue
                counts[name] += 1
        entries: list[GlossaryEntry] = []
        for name, count in counts.most_common():
            if count < self._minimum_count:
                continue
            normalize_aliases, normalize_canonical = safe_normalization_flags(name)
            entries.append(
                GlossaryEntry(
                    canonical=name,
                    category="character",
                    normalize_aliases=normalize_aliases,
                    normalize_canonical=normalize_canonical,
                )
            )
        return tuple(entries)

    def enrich(self, profile: SeriesProfile, cues: Iterable[Cue]) -> SeriesProfile:
        extra = self.extract(cues)
        return self._composer.merge(profile, extra) if extra else profile
