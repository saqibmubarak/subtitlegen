from __future__ import annotations

import re
from collections.abc import Callable

from subtitlegen.profiles.models import SeriesProfile


def _literal_replacement(value: str) -> Callable[[re.Match[str]], str]:
    def replace(_match: re.Match[str]) -> str:
        return value

    return replace


class GlossaryNormalizer:
    """Replace complete aliases with canonical spellings without touching substrings."""

    def normalize(self, text: str, profile: SeriesProfile) -> str:
        replacements: list[tuple[str, str]] = []
        for entry in profile.terms:
            if entry.normalize_aliases:
                replacements.extend((spelling, entry.canonical) for spelling in entry.aliases)
            if entry.normalize_canonical:
                replacements.append((entry.canonical, entry.canonical))
        replacements.sort(key=lambda item: len(item[0]), reverse=True)

        normalized = text
        for spelling, canonical in replacements:
            pattern = re.compile(rf"(?<!\w){re.escape(spelling)}(?!\w)", re.IGNORECASE)
            normalized = pattern.sub(_literal_replacement(canonical), normalized)
        return normalized
