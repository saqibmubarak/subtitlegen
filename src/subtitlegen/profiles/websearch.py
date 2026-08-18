from __future__ import annotations

import logging
import re
from html import unescape
from urllib.parse import urlencode

from subtitlegen.profiles.composer import safe_normalization_flags
from subtitlegen.profiles.http import HttpClient, UrllibHttpClient
from subtitlegen.profiles.models import GlossaryEntry

logger = logging.getLogger(__name__)

_SEARCH = "https://html.duckduckgo.com/html/"
_SNIPPET = re.compile(
    r'class="result__(?:snippet|a)"[^>]*>(.*?)</(?:a|td|span|div)>',
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")
_NAME = re.compile(r"\b([A-Z][A-Za-z]+(?:[ '.-][A-Z][A-Za-z]+){0,3})\b")
_STOP = frozenset(
    {
        "characters",
        "duckduckgo",
        "english",
        "official",
        "wikipedia",
        "wikia",
        "fandom",
    }
)


class WebSearchGlossarySource:
    """Extract candidate names from a fast HTML search page when Wikipedia is thin."""

    def __init__(self, http: HttpClient | None = None) -> None:
        self._http = http or UrllibHttpClient()

    def terms(self, title: str) -> tuple[GlossaryEntry, ...]:
        query = f"{title} characters places glossary"
        url = f"{_SEARCH}?{urlencode({'q': query})}"
        try:
            html = self._http.get_text(url)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            logger.warning("web search failed: %s", error)
            return ()
        names: list[str] = []
        for snippet in _SNIPPET.findall(html):
            text = unescape(_TAG.sub(" ", snippet))
            names.extend(match.group(1) for match in _NAME.finditer(text))
        seen: set[str] = set()
        entries: list[GlossaryEntry] = []
        for name in names:
            key = name.casefold()
            if key in seen or key in _STOP or len(name) < 3:
                continue
            seen.add(key)
            normalize_aliases, normalize_canonical = safe_normalization_flags(name)
            entries.append(
                GlossaryEntry(
                    canonical=name,
                    category="character",
                    normalize_aliases=normalize_aliases,
                    normalize_canonical=normalize_canonical,
                )
            )
            if len(entries) >= 40:
                break
        return tuple(entries)
