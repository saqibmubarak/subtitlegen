"""Versioned series context and terminology support."""

from subtitlegen.profiles.builder import AutomaticProfileBuilder
from subtitlegen.profiles.identity import MediaIdentity, PathIdentityInferencer
from subtitlegen.profiles.models import GlossaryEntry, SeriesProfile
from subtitlegen.profiles.repository import ProfileRepository
from subtitlegen.profiles.resolver import ProfileResolver, ResolvedProfile
from subtitlegen.profiles.selector import ContextSelector

__all__ = [
    "AutomaticProfileBuilder",
    "ContextSelector",
    "GlossaryEntry",
    "MediaIdentity",
    "PathIdentityInferencer",
    "ProfileRepository",
    "ProfileResolver",
    "ResolvedProfile",
    "SeriesProfile",
]
