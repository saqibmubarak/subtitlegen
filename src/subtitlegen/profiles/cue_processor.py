from __future__ import annotations

from collections.abc import Iterable

from subtitlegen.domain.models import Cue
from subtitlegen.profiles.correction import ConfidenceGatedCorrector
from subtitlegen.profiles.models import SeriesProfile
from subtitlegen.profiles.normalizer import GlossaryNormalizer


class ProfileCueProcessor:
    def __init__(
        self,
        profile: SeriesProfile,
        normalizer: GlossaryNormalizer,
        corrector: ConfidenceGatedCorrector | None = None,
    ) -> None:
        self._profile = profile
        self._normalizer = normalizer
        self._corrector = corrector

    def process(self, cues: Iterable[Cue]) -> list[Cue]:
        return [
            Cue(
                start=cue.start,
                end=cue.end,
                text=self._process_text(cue),
                confidence=cue.confidence,
            )
            for cue in cues
        ]

    def _process_text(self, cue: Cue) -> str:
        if self._corrector is None or cue.confidence is None:
            return self._normalizer.normalize(cue.text, self._profile)
        return self._corrector.correct(
            cue.text,
            confidence=cue.confidence,
            profile=self._profile,
        ).output
