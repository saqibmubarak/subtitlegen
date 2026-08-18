from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from subtitlegen.profiles.builder import AutomaticProfileBuilder
from subtitlegen.profiles.identity import MediaIdentity, PathIdentityInferencer
from subtitlegen.profiles.models import SeriesProfile
from subtitlegen.profiles.repository import ProfileRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResolvedProfile:
    profile: SeriesProfile | None
    identity: MediaIdentity | None
    source: str
    enable_visual: bool


class ProfileResolver:
    """Resolve a profile from an explicit id, shipped YAML, cache, or automatic sources."""

    def __init__(
        self,
        cache: ProfileRepository,
        shipped: ProfileRepository | None = None,
        inferencer: PathIdentityInferencer | None = None,
        builder: AutomaticProfileBuilder | None = None,
    ) -> None:
        self._cache = cache
        self._shipped = shipped
        self._inferencer = inferencer or PathIdentityInferencer()
        self._builder = builder or AutomaticProfileBuilder()

    def resolve(
        self,
        paths: Sequence[Path],
        *,
        explicit_id: str | None = None,
        auto: bool = True,
        explicit_repository: ProfileRepository | None = None,
    ) -> ResolvedProfile:
        if explicit_id:
            repository = explicit_repository or self._shipped or self._cache
            profile = repository.load(explicit_id)
            return ResolvedProfile(
                profile=profile,
                identity=MediaIdentity(profile.title, profile.profile_id),
                source="explicit",
                enable_visual=bool(profile.visual_translations),
            )
        identity = self._infer(paths)
        if identity is None:
            return ResolvedProfile(None, None, "none", False)
        shipped = self._load_matching(self._shipped, identity)
        if shipped is not None:
            return ResolvedProfile(
                shipped,
                identity,
                "shipped",
                enable_visual=bool(shipped.visual_translations),
            )
        cached = self._load_matching(self._cache, identity)
        if cached is not None:
            return ResolvedProfile(
                cached,
                identity,
                "cache",
                enable_visual=bool(cached.visual_translations),
            )
        if not auto:
            return ResolvedProfile(None, identity, "none", False)
        built = self._builder.build(identity)
        self._cache.save(built.profile)
        logger.info(
            "created %s profile %s with %d terms from %s",
            built.profile.profile_id,
            built.profile.title,
            len(built.profile.terms),
            built.source,
        )
        return ResolvedProfile(
            built.profile,
            identity,
            built.source,
            built.enable_visual,
        )

    def _infer(self, paths: Sequence[Path]) -> MediaIdentity | None:
        for path in paths:
            identity = self._inferencer.infer(path)
            if identity is not None:
                return identity
        return None

    @staticmethod
    def _load_matching(
        repository: ProfileRepository | None,
        identity: MediaIdentity,
    ) -> SeriesProfile | None:
        if repository is None:
            return None
        try:
            available = repository.available()
        except OSError:
            return None
        if identity.profile_id in available:
            return repository.load(identity.profile_id)
        for profile_id in available:
            profile = repository.load(profile_id)
            if ProfileResolver._titles_match(identity.title, profile.title):
                return profile
        return None

    @staticmethod
    def _titles_match(inferred: str, known: str) -> bool:
        left = inferred.casefold()
        right = known.casefold()
        if left == right:
            return True
        shorter, longer = sorted((left, right), key=len)
        return len(shorter) >= 5 and shorter in longer
