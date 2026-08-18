from __future__ import annotations

from subtitlegen.profiles.identity import slugify
from subtitlegen.profiles.models import GlossaryEntry, SeriesProfile

_UNSAFE_SINGLE = frozenset(
    {
        "ace",
        "brook",
        "captain",
        "city",
        "doctor",
        "earth",
        "fire",
        "frank",
        "girl",
        "guy",
        "island",
        "king",
        "law",
        "lord",
        "man",
        "marine",
        "nation",
        "pirate",
        "queen",
        "robin",
        "sea",
        "state",
        "sunny",
        "tribe",
        "water",
        "world",
    }
)


def safe_normalization_flags(canonical: str) -> tuple[bool, bool]:
    token = canonical.strip()
    if " " in token or any(marker in token for marker in ("'", ".", "-")):
        return True, True
    if token.casefold() in _UNSAFE_SINGLE:
        return False, False
    return True, True


class ProfileComposer:
    """Merge glossary sources without allowing spelling collisions."""

    def compose(
        self,
        title: str,
        sources: tuple[tuple[GlossaryEntry, ...], ...],
        visual_translations: tuple[tuple[str, str], ...] = (),
        *,
        language: str = "en",
        profile_id: str | None = None,
    ) -> SeriesProfile:
        owners: dict[str, str] = {}
        terms: list[GlossaryEntry] = []
        for group in sources:
            for entry in group:
                if self._conflicts(entry, owners):
                    continue
                for spelling in (entry.canonical, *entry.aliases):
                    owners[spelling.casefold()] = entry.canonical
                terms.append(entry)
        translations: list[tuple[str, str]] = []
        seen_sources: set[str] = set()
        for source, target in visual_translations:
            if source in seen_sources or not source.strip() or not target.strip():
                continue
            seen_sources.add(source)
            translations.append((source, target))
        return SeriesProfile(
            schema_version=1,
            profile_id=profile_id or slugify(title),
            title=title,
            language=language,
            terms=tuple(terms),
            visual_translations=tuple(translations),
        )

    def merge(self, base: SeriesProfile, extra: tuple[GlossaryEntry, ...]) -> SeriesProfile:
        return self.compose(
            base.title,
            (base.terms, extra),
            base.visual_translations,
            language=base.language,
            profile_id=base.profile_id,
        )

    @staticmethod
    def _conflicts(entry: GlossaryEntry, owners: dict[str, str]) -> bool:
        for spelling in (entry.canonical, *entry.aliases):
            owner = owners.get(spelling.casefold())
            if owner is not None and owner != entry.canonical:
                return True
        return False
