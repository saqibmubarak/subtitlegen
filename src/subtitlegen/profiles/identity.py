from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MediaIdentity:
    title: str
    profile_id: str
    arc: str | None = None
    episode: str | None = None

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.profile_id.strip():
            raise ValueError("media identity title and profile id must not be blank")


_BRACKET_TAG = re.compile(r"\[(?:[^\]]+)\]|\((?:19|20)\d{2}\)")
_SEASON_EPISODE = re.compile(
    r"^(?P<title>.+?)[\s._-]+[sS](?P<season>\d{1,2})[eE](?P<episode>\d{1,3})\b"
    r"(?:[\s._-]+(?P<rest>.+))?$"
)
_EPISODE_WORD = re.compile(
    r"^(?P<title>.+?)[\s._-]+(?:[eE](?:p(?:isode)?)?\.?[\s._-]*)(?P<episode>\d{1,4})\b"
    r"(?:[\s._-]+(?P<rest>.+))?$"
)
_DASH_EPISODE = re.compile(
    r"^(?P<title>.+?)[\s._-]+(?P<episode>\d{2,4})(?:[\s._-]+(?P<rest>.+))?$"
)
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")
_NON_SLUG = re.compile(r"[^a-z0-9]+")
_JUNK_TITLES = frozenset(
    {
        "audio",
        "clip",
        "data",
        "download",
        "downloads",
        "episode",
        "episodes",
        "input",
        "media",
        "movie",
        "movies",
        "output",
        "sample",
        "season",
        "temp",
        "tmp",
        "video",
        "videos",
    }
)
_JUNK_PARENTS = frozenset(
    {
        "data",
        "downloads",
        "home",
        "media",
        "mnt",
        "private",
        "tmp",
        "users",
        "var",
        "volumes",
    }
)
_ARC_STOP = frozenset(
    {
        "complete",
        "dub",
        "dubbed",
        "english",
        "final",
        "hd",
        "japanese",
        "multi",
        "remux",
        "sub",
        "uncensored",
    }
)


def slugify(title: str) -> str:
    slug = _NON_SLUG.sub("-", title.casefold()).strip("-")
    if not slug:
        raise ValueError("title does not produce a profile id")
    return slug


class PathIdentityInferencer:
    """Read a series title, episode, and optional arc from a file or directory name."""

    def infer(self, path: Path) -> MediaIdentity | None:
        resolved = path.expanduser()
        if resolved.is_file() or resolved.suffix:
            return self._from_file(resolved)
        return self._from_directory(resolved)

    def _from_directory(self, path: Path) -> MediaIdentity | None:
        title = self._clean_title(path.name)
        if not self._usable_title(title) or self._junk_parent(path):
            return None
        return MediaIdentity(title=title, profile_id=slugify(title))

    def _from_file(self, path: Path) -> MediaIdentity | None:
        stem = _BRACKET_TAG.sub(" ", path.stem)
        parsed = self._parse_filename(stem)
        title = parsed[0] if parsed is not None else None
        episode = parsed[1] if parsed is not None else None
        rest = parsed[2] if parsed is not None else None
        if title is None or not self._usable_title(title):
            parent_title = self._clean_title(path.parent.name)
            if self._usable_title(parent_title) and not self._junk_parent(path.parent):
                title = parent_title
            else:
                return None
        arc = self._arc_from_rest(rest)
        if arc is None:
            parent = self._clean_title(path.parent.name)
            if (
                self._usable_title(parent)
                and parent.casefold() != title.casefold()
                and not self._junk_parent(path.parent)
            ):
                arc = parent
        return MediaIdentity(
            title=title,
            profile_id=slugify(title),
            arc=arc,
            episode=episode,
        )

    def _parse_filename(self, stem: str) -> tuple[str, str | None, str | None] | None:
        normalized = stem.replace("_", " ").strip()
        dotted = normalized.replace(".", " ")
        for candidate in (normalized, dotted):
            for pattern in (_SEASON_EPISODE, _EPISODE_WORD, _DASH_EPISODE):
                match = pattern.match(candidate)
                if match is None:
                    continue
                title = self._clean_title(match.group("title"))
                if not self._usable_title(title):
                    continue
                rest = match.groupdict().get("rest")
                return title, str(int(match.group("episode"))), rest
        cleaned = self._clean_title(dotted)
        if self._usable_title(cleaned):
            return cleaned, None, None
        return None

    @staticmethod
    def _clean_title(value: str) -> str:
        spaced = _CAMEL.sub(" ", value.replace("_", " ").replace(".", " "))
        return re.sub(r"\s+", " ", spaced).strip(" -")

    @staticmethod
    def _usable_title(title: str) -> bool:
        key = title.casefold()
        if len(title) < 3 or key in _JUNK_TITLES:
            return False
        if key.startswith("test_") or key.startswith("pytest"):
            return False
        return any(character.isalpha() for character in title)

    @staticmethod
    def _junk_parent(path: Path) -> bool:
        return path.name.casefold() in _JUNK_PARENTS or path.name.startswith("test_")

    @staticmethod
    def _arc_from_rest(rest: str | None) -> str | None:
        if rest is None:
            return None
        cleaned = PathIdentityInferencer._clean_title(rest)
        words = [word for word in cleaned.split() if word.casefold() not in _ARC_STOP]
        if not words or len(words) > 3:
            return None
        if any(not word[:1].isalpha() for word in words):
            return None
        return " ".join(words)
