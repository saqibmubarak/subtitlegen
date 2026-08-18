from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from subtitlegen.profiles.models import GlossaryEntry, SeriesProfile


class ProfileRepository:
    def __init__(self, root: Path) -> None:
        self._root = root

    def load(self, profile_id: str) -> SeriesProfile:
        if not profile_id or Path(profile_id).name != profile_id:
            raise ValueError("profile id must be a simple file name")
        path = self._root / f"{profile_id}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"series profile not found: {path}")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            profile = self._decode(data)
            if profile.profile_id != profile_id:
                raise ValueError("profile id does not match its file name")
            return profile
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
            raise RuntimeError(f"invalid series profile: {path}") from error

    def available(self) -> tuple[str, ...]:
        if not self._root.is_dir():
            return ()
        return tuple(sorted(path.stem for path in self._root.glob("*.yaml") if path.is_file()))

    def save(self, profile: SeriesProfile) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{profile.profile_id}.yaml"
        path.write_text(
            yaml.safe_dump(self.encode(profile), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def encode(profile: SeriesProfile) -> dict[str, object]:
        terms: list[dict[str, object]] = []
        for entry in profile.terms:
            item: dict[str, object] = {
                "canonical": entry.canonical,
                "category": entry.category,
            }
            if entry.aliases:
                item["aliases"] = list(entry.aliases)
            if entry.arcs:
                item["arcs"] = list(entry.arcs)
            if entry.episodes:
                item["episodes"] = list(entry.episodes)
            if not entry.normalize_aliases:
                item["normalize_aliases"] = False
            if not entry.normalize_canonical:
                item["normalize_canonical"] = False
            terms.append(item)
        payload: dict[str, object] = {
            "schema_version": profile.schema_version,
            "profile_id": profile.profile_id,
            "title": profile.title,
            "language": profile.language,
            "terms": terms,
        }
        if profile.visual_translations:
            payload["visual_translations"] = {
                source: target for source, target in profile.visual_translations
            }
        return payload

    @classmethod
    def default(cls, search_roots: tuple[Path, ...] | None = None) -> ProfileRepository:
        roots = search_roots or (
            Path.cwd() / "profiles",
            Path(sys.prefix) / "share" / "subtitlegen" / "profiles",
        )
        for root in roots:
            if root.is_dir() and any(root.glob("*.yaml")):
                return cls(root)
        raise FileNotFoundError("no installed series profiles were found")

    @staticmethod
    def _decode(data: Any) -> SeriesProfile:
        if not isinstance(data, dict) or not isinstance(data.get("terms"), list):
            raise ValueError("profile must be a mapping with a terms list")
        allowed_profile_fields = {
            "schema_version",
            "profile_id",
            "title",
            "language",
            "terms",
            "visual_translations",
        }
        unknown_profile_fields = set(data) - allowed_profile_fields
        if unknown_profile_fields:
            raise ValueError(f"unknown profile fields: {sorted(unknown_profile_fields)}")

        allowed_term_fields = {
            "canonical",
            "aliases",
            "category",
            "arcs",
            "episodes",
            "normalize_aliases",
            "normalize_canonical",
        }
        terms: list[GlossaryEntry] = []
        for item in data["terms"]:
            if not isinstance(item, dict):
                raise ValueError("each glossary term must be a mapping")
            unknown_term_fields = set(item) - allowed_term_fields
            if unknown_term_fields:
                raise ValueError(f"unknown glossary fields: {sorted(unknown_term_fields)}")
            canonical = item.get("canonical")
            category = item.get("category", "term")
            if not isinstance(canonical, str) or not isinstance(category, str):
                raise ValueError("canonical and category fields must be strings")
            normalize_aliases = item.get("normalize_aliases", True)
            normalize_canonical = item.get("normalize_canonical", True)
            if not isinstance(normalize_aliases, bool) or not isinstance(
                normalize_canonical, bool
            ):
                raise ValueError("normalization flags must be booleans")
            terms.append(
                GlossaryEntry(
                    canonical=canonical,
                    aliases=ProfileRepository._string_tuple(item.get("aliases", [])),
                    category=category,
                    arcs=ProfileRepository._string_tuple(item.get("arcs", [])),
                    episodes=ProfileRepository._string_tuple(
                        item.get("episodes", []), allow_numbers=True
                    ),
                    normalize_aliases=normalize_aliases,
                    normalize_canonical=normalize_canonical,
                )
            )
        identity = (data.get("profile_id"), data.get("title"), data.get("language", "en"))
        if not all(isinstance(value, str) for value in identity):
            raise ValueError("profile identity fields must be strings")
        return SeriesProfile(
            schema_version=data["schema_version"],
            profile_id=data["profile_id"],
            title=data["title"],
            language=data.get("language", "en"),
            terms=tuple(terms),
            visual_translations=ProfileRepository._translations(
                data.get("visual_translations", {})
            ),
        )

    @staticmethod
    def _string_tuple(value: Any, *, allow_numbers: bool = False) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise ValueError("glossary collection fields must be lists")
        allowed = (str, int) if allow_numbers else (str,)
        if any(not isinstance(item, allowed) for item in value):
            raise ValueError("glossary collection values have invalid types")
        return tuple(str(item) for item in value)

    @staticmethod
    def _translations(value: Any) -> tuple[tuple[str, str], ...]:
        if not isinstance(value, dict) or any(
            not isinstance(source, str) or not isinstance(target, str)
            for source, target in value.items()
        ):
            raise ValueError("visual_translations must map strings to strings")
        return tuple((source, target) for source, target in value.items())
