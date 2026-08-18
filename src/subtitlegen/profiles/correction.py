from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

from subtitlegen.profiles.models import SeriesProfile
from subtitlegen.profiles.normalizer import GlossaryNormalizer


class LocalTextCorrector(Protocol):
    def correct(self, text: str, *, glossary: tuple[str, ...]) -> str:
        """Correct uncertain text using a local-only implementation."""


class ConservativeLocalCorrector:
    """Correct close single-token glossary misspellings in low-confidence cues."""

    def __init__(self, similarity_threshold: float = 0.8) -> None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity threshold must be between zero and one")
        self._threshold = similarity_threshold

    def correct(self, text: str, *, glossary: tuple[str, ...]) -> str:
        candidates = tuple(term for term in glossary if " " not in term)

        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            eligible = (
                term
                for term in candidates
                if term[:1].casefold() == token[:1].casefold()
            )
            scored = [
                (
                    SequenceMatcher(None, token.casefold(), term.casefold()).ratio(),
                    term,
                )
                for term in eligible
            ]
            if not scored:
                return token
            score, canonical = max(scored)
            return canonical if score >= self._threshold else token

        return re.sub(r"(?<!\w)[A-Za-z]+(?:-[A-Za-z]+)*(?!\w)", replace, text)


@dataclass(frozen=True, slots=True)
class CorrectionDecision:
    original: str
    output: str
    applied: bool
    reason: str


class ConfidenceGatedCorrector:
    def __init__(
        self,
        normalizer: GlossaryNormalizer,
        corrector: LocalTextCorrector | None = None,
        *,
        confidence_threshold: float = 0.8,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence threshold must be between zero and one")
        self._normalizer = normalizer
        self._corrector = corrector
        self._threshold = confidence_threshold

    def correct(
        self,
        text: str,
        *,
        confidence: float,
        profile: SeriesProfile,
    ) -> CorrectionDecision:
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        normalized = self._normalizer.normalize(text, profile)
        glossary_changed = normalized != text

        if confidence >= self._threshold:
            return CorrectionDecision(
                original=text,
                output=normalized,
                applied=glossary_changed,
                reason="glossary" if glossary_changed else "high-confidence",
            )
        if self._corrector is None:
            return CorrectionDecision(
                original=text,
                output=normalized,
                applied=glossary_changed,
                reason="glossary" if glossary_changed else "no-local-corrector",
            )

        glossary = tuple(
            entry.canonical
            for entry in profile.terms
            if entry.normalize_aliases and entry.normalize_canonical
        )
        candidate = self._corrector.correct(normalized, glossary=glossary).strip()
        output = candidate or normalized
        return CorrectionDecision(
            original=text,
            output=output,
            applied=output != text,
            reason="low-confidence-local",
        )
