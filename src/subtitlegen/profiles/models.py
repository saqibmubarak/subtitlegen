from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    canonical: str
    aliases: tuple[str, ...] = ()
    category: str = "term"
    arcs: tuple[str, ...] = ()
    episodes: tuple[str, ...] = ()
    normalize_aliases: bool = True
    normalize_canonical: bool = True

    def __post_init__(self) -> None:
        if not self.canonical.strip():
            raise ValueError("canonical glossary term must not be blank")
        if any(not alias.strip() for alias in self.aliases):
            raise ValueError("glossary aliases must not be blank")
        if not self.category.strip():
            raise ValueError("glossary category must not be blank")
        if not isinstance(self.normalize_aliases, bool) or not isinstance(
            self.normalize_canonical, bool
        ):
            raise ValueError("normalization flags must be booleans")

    def applies_to(self, *, arc: str | None = None, episode: str | None = None) -> bool:
        if self.arcs and arc is not None and arc.casefold() not in {
            item.casefold() for item in self.arcs
        }:
            return False
        return not (
            self.episodes
            and episode is not None
            and episode.casefold() not in {item.casefold() for item in self.episodes}
        )


@dataclass(frozen=True, slots=True)
class SeriesProfile:
    schema_version: int
    profile_id: str
    title: str
    language: str
    terms: tuple[GlossaryEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported profile schema {self.schema_version}")
        if not self.profile_id or not self.title or not self.language:
            raise ValueError("profile identity fields must not be blank")
        owners: dict[str, str] = {}
        for entry in self.terms:
            for spelling in (entry.canonical, *entry.aliases):
                key = spelling.casefold()
                owner = owners.get(key)
                if owner is not None and owner != entry.canonical:
                    raise ValueError(
                        f"glossary spelling '{spelling}' collides between '{owner}' "
                        f"and '{entry.canonical}'"
                    )
                owners[key] = entry.canonical
