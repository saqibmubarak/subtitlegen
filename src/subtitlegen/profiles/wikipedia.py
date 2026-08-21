from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from subtitlegen.profiles.composer import safe_normalization_flags
from subtitlegen.profiles.http import HttpClient, UrllibHttpClient
from subtitlegen.profiles.models import GlossaryEntry

logger = logging.getLogger(__name__)

_API = "https://en.wikipedia.org/w/api.php"
_BOLD = re.compile(r"'''([^']{2,50})'''")
_LINK = re.compile(r"\[\[(?:File:|Image:|Category:)[^\]]*\]\]|\[\[(?:[^|\]]+\|)?([^\]]{2,50})\]\]")
_INFOBOX_NAME = re.compile(
    r"\|\s*(?:protagonist|characters?|creator|location|setting)\s*=\s*(.+)",
    re.IGNORECASE,
)
_ANIME_MARKERS = (
    "anime",
    "manga",
    "japanese television",
    "japanese films",
    "original video animation",
)
_STOP = frozenset(
    {
        "anime",
        "cast",
        "character",
        "characters",
        "episode",
        "episodes",
        "external links",
        "film",
        "list",
        "manga",
        "plot",
        "reception",
        "references",
        "see also",
        "season",
        "series",
        "television",
        "wikipedia",
    }
)


@dataclass(frozen=True, slots=True)
class WikipediaDocument:
    title: str
    extract: str
    wikitext: str
    categories: tuple[str, ...]
    japanese_title: str | None
    character_wikitext: str


class WikipediaGlossarySource:
    """Build glossary terms from Wikipedia extracts and character-list pages."""

    def __init__(self, http: HttpClient | None = None) -> None:
        self._http = http or UrllibHttpClient()

    def fetch(self, query: str) -> WikipediaDocument | None:
        title = self._opensearch(query)
        if title is None:
            return None
        with ThreadPoolExecutor(max_workers=4) as pool:
            page_future = pool.submit(self._page, title)
            characters_future = pool.submit(self._named_list, title, "characters")
            locations_future = pool.submit(self._named_list, title, "locations")
            page = page_future.result()
            character_wikitext = characters_future.result()
            location_wikitext = locations_future.result()
        if page is None:
            return None
        extract, wikitext, categories, japanese_title = page
        return WikipediaDocument(
            title=title,
            extract=extract,
            wikitext=wikitext,
            categories=categories,
            japanese_title=japanese_title,
            character_wikitext="\n".join(
                part for part in (character_wikitext, location_wikitext) if part
            ),
        )

    def terms(self, document: WikipediaDocument) -> tuple[GlossaryEntry, ...]:
        names: list[tuple[str, str]] = []
        names.extend((name, "character") for name in self._infobox_names(document.wikitext))
        names.extend((name, "character") for name in self._bold_names(document.character_wikitext))
        names.extend(
            (name, "character") for name in self._list_link_names(document.character_wikitext)
        )
        names.extend((name, "character") for name in self._bold_names(document.wikitext))
        names.extend((name, "term") for name in self._proper_names(document.extract))
        seen: set[str] = set()
        entries: list[GlossaryEntry] = []
        for name, category in names:
            key = name.casefold()
            if key in seen or not self._usable(name):
                continue
            seen.add(key)
            normalize_aliases, normalize_canonical = safe_normalization_flags(name)
            entries.append(
                GlossaryEntry(
                    canonical=name,
                    category=category,
                    normalize_aliases=normalize_aliases,
                    normalize_canonical=normalize_canonical,
                )
            )
            if len(entries) >= 400:
                break
        return tuple(entries)

    def looks_like_anime(self, document: WikipediaDocument) -> bool:
        haystack = " ".join((document.extract, *document.categories)).casefold()
        return document.japanese_title is not None or any(
            marker in haystack for marker in _ANIME_MARKERS
        )

    def _opensearch(self, query: str) -> str | None:
        payload = self._get(
            {
                "action": "opensearch",
                "search": query,
                "limit": "3",
                "namespace": "0",
                "format": "json",
            }
        )
        if not isinstance(payload, list) or len(payload) < 2:
            return None
        titles = payload[1]
        if not isinstance(titles, list) or not titles:
            return None
        first = titles[0]
        return first if isinstance(first, str) and first.strip() else None

    def _page(
        self, title: str
    ) -> tuple[str, str, tuple[str, ...], str | None] | None:
        payload = self._get(
            {
                "action": "query",
                "prop": "extracts|categories|langlinks|revisions",
                "explaintext": "1",
                "exintro": "1",
                "exchars": "1800",
                "cllimit": "20",
                "lllang": "ja",
                "rvprop": "content",
                "rvslots": "main",
                "titles": title,
                "format": "json",
                "redirects": "1",
            }
        )
        page = self._first_page(payload)
        if page is None:
            return None
        extract = page.get("extract")
        if not isinstance(extract, str):
            extract = ""
        categories = tuple(
            str(item["title"]).removeprefix("Category:")
            for item in page.get("categories", [])
            if isinstance(item, dict) and isinstance(item.get("title"), str)
        )
        japanese = None
        langlinks = page.get("langlinks")
        if isinstance(langlinks, list) and langlinks:
            first = langlinks[0]
            if isinstance(first, dict) and isinstance(first.get("*") or first.get("title"), str):
                japanese = str(first.get("*") or first.get("title"))
        wikitext = ""
        revisions = page.get("revisions")
        if isinstance(revisions, list) and revisions:
            revision = revisions[0]
            if isinstance(revision, dict):
                slots = revision.get("slots")
                if isinstance(slots, dict):
                    main = slots.get("main")
                    if isinstance(main, dict) and isinstance(main.get("*"), str):
                        wikitext = main["*"][:80_000]
                elif isinstance(revision.get("*"), str):
                    wikitext = str(revision["*"])[:80_000]
        return extract, wikitext, categories, japanese

    def _named_list(self, title: str, kind: str) -> str:
        payload = self._get(
            {
                "action": "query",
                "list": "search",
                "srsearch": f"List of {title} {kind}",
                "srlimit": "1",
                "format": "json",
            }
        )
        if not isinstance(payload, dict):
            return ""
        results = payload.get("query", {}).get("search", [])
        if not isinstance(results, list) or not results:
            return ""
        first = results[0]
        if not isinstance(first, dict) or not isinstance(first.get("title"), str):
            return ""
        haystack = first["title"].casefold()
        if kind.rstrip("s") not in haystack and kind not in haystack:
            return ""
        page = self._page(first["title"])
        return page[1] if page is not None else ""

    def _character_list(self, title: str) -> str:
        return self._named_list(title, "characters")

    def _get(self, params: dict[str, str]) -> Any:
        url = f"{_API}?{urlencode(params)}"
        try:
            return self._http.get_json(url)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            logger.warning("wikipedia request failed: %s", error)
            return None

    @staticmethod
    def _first_page(payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        pages = payload.get("query", {}).get("pages")
        if not isinstance(pages, dict) or not pages:
            return None
        page = next(iter(pages.values()))
        return page if isinstance(page, dict) and "missing" not in page else None

    @classmethod
    def _infobox_names(cls, wikitext: str) -> tuple[str, ...]:
        names: list[str] = []
        for match in _INFOBOX_NAME.finditer(wikitext):
            names.extend(cls._link_names(match.group(1)))
        return tuple(names)

    @classmethod
    def _bold_names(cls, wikitext: str) -> tuple[str, ...]:
        return tuple(match.group(1).strip() for match in _BOLD.finditer(wikitext))

    @classmethod
    def _list_link_names(cls, wikitext: str) -> tuple[str, ...]:
        names: list[str] = []
        for line in wikitext.splitlines():
            stripped = line.lstrip()
            if not stripped.startswith(("*", "#", ":")):
                continue
            names.extend(cls._link_names(stripped))
            names.extend(cls._bold_names(stripped))
        return tuple(names)

    @classmethod
    def _link_names(cls, value: str) -> tuple[str, ...]:
        return tuple(
            match.group(1).strip()
            for match in _LINK.finditer(value)
            if match.group(1) is not None
        )

    @staticmethod
    def _proper_names(extract: str) -> tuple[str, ...]:
        return tuple(
            match.group(0)
            for match in re.finditer(r"\b([A-Z][A-Za-z]+(?:[ '.-][A-Z][A-Za-z]+){0,3})\b", extract)
        )

    @staticmethod
    def _usable(name: str) -> bool:
        cleaned = name.strip()
        key = cleaned.casefold()
        if len(cleaned) < 3 or key in _STOP:
            return False
        if cleaned.startswith(("File:", "Image:", "Category:")):
            return False
        if not any(character.isalpha() for character in cleaned):
            return False
        words = cleaned.split()
        return all(word[:1].isalpha() for word in words)
