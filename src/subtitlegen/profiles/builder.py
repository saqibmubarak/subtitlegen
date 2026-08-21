from __future__ import annotations

import logging
from dataclasses import dataclass

from subtitlegen.profiles.composer import ProfileComposer
from subtitlegen.profiles.identity import MediaIdentity
from subtitlegen.profiles.models import SeriesProfile
from subtitlegen.profiles.websearch import WebSearchGlossarySource
from subtitlegen.profiles.wikipedia import WikipediaGlossarySource

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BuiltProfile:
    profile: SeriesProfile
    enable_visual: bool
    source: str


class AutomaticProfileBuilder:
    """Create a series profile from Wikipedia, web search, and the inferred title."""

    def __init__(
        self,
        wikipedia: WikipediaGlossarySource | None = None,
        web_search: WebSearchGlossarySource | None = None,
        composer: ProfileComposer | None = None,
        minimum_remote_terms: int = 8,
    ) -> None:
        if minimum_remote_terms < 1:
            raise ValueError("minimum remote term count must be positive")
        self._wikipedia = wikipedia or WikipediaGlossarySource()
        self._web_search = web_search or WebSearchGlossarySource()
        self._composer = composer or ProfileComposer()
        self._minimum_remote_terms = minimum_remote_terms

    def build(self, identity: MediaIdentity) -> BuiltProfile:
        document = self._wikipedia.fetch(identity.title)
        wiki_terms = self._wikipedia.terms(document) if document is not None else ()
        search_terms = (
            self._web_search.terms(identity.title)
            if len(wiki_terms) < self._minimum_remote_terms
            else ()
        )
        title = document.title if document is not None else identity.title
        profile = self._composer.compose(
            title,
            (wiki_terms, search_terms),
            profile_id=identity.profile_id,
        )
        source = "wikipedia" if wiki_terms else "search" if search_terms else "local"
        if not profile.terms:
            logger.info("built empty remote glossary for %s; using title-only profile", title)
        enable_visual = document is not None and self._wikipedia.looks_like_anime(document)
        return BuiltProfile(profile=profile, enable_visual=enable_visual, source=source)

    def enrich(self, profile: SeriesProfile) -> SeriesProfile:
        document = self._wikipedia.fetch(profile.title)
        wiki_terms = self._wikipedia.terms(document) if document is not None else ()
        search_terms = (
            self._web_search.terms(profile.title)
            if len(wiki_terms) < self._minimum_remote_terms
            else ()
        )
        if not wiki_terms and not search_terms:
            return profile
        return self._composer.compose(
            profile.title,
            (profile.terms, wiki_terms, search_terms),
            profile.visual_translations,
            language=profile.language,
            profile_id=profile.profile_id,
        )
