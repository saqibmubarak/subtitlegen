import pytest

from subtitlegen.domain.models import Cue
from subtitlegen.profiles.correction import (
    ConfidenceGatedCorrector,
    ConservativeLocalCorrector,
    CorrectionDecision,
)
from subtitlegen.profiles.cue_processor import ProfileCueProcessor
from subtitlegen.profiles.models import GlossaryEntry, SeriesProfile
from subtitlegen.profiles.normalizer import GlossaryNormalizer


class FakeLocalCorrector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def correct(self, text: str, *, glossary: tuple[str, ...]) -> str:
        self.calls.append((text, glossary))
        return text.replace("isle", "Avatar")


def _profile() -> SeriesProfile:
    return SeriesProfile(
        1,
        "avatar",
        "Avatar",
        "en",
        (
            GlossaryEntry("Aang", aliases=("Ang",), category="character"),
            GlossaryEntry("airbender", aliases=("air bender",)),
            GlossaryEntry(
                "Haki",
                aliases=("hockey",),
                normalize_aliases=False,
            ),
            GlossaryEntry("Buggy"),
            GlossaryEntry(
                "Law",
                normalize_aliases=False,
                normalize_canonical=False,
            ),
        ),
    )


def test_normalizer_handles_aliases_case_punctuation_and_substrings() -> None:
    normalizer = GlossaryNormalizer()
    text = normalizer.normalize("ANG is an air bender; danger is unchanged.", _profile())
    assert text == "Aang is an airbender; danger is unchanged."
    assert normalizer.normalize("They played hockey.", _profile()) == "They played hockey."


def test_confidence_gate_protects_high_confidence_text() -> None:
    local = FakeLocalCorrector()
    corrector = ConfidenceGatedCorrector(GlossaryNormalizer(), local)
    decision = corrector.correct("This is clear.", confidence=0.95, profile=_profile())
    assert decision == CorrectionDecision(
        "This is clear.",
        "This is clear.",
        False,
        "high-confidence",
    )
    assert local.calls == []
    with pytest.raises(ValueError):
        ConfidenceGatedCorrector(GlossaryNormalizer(), confidence_threshold=2)


def test_confidence_gate_normalizes_glossary_and_calls_local_only_when_low() -> None:
    local = FakeLocalCorrector()
    corrector = ConfidenceGatedCorrector(GlossaryNormalizer(), local)
    glossary = corrector.correct("Ang arrived.", confidence=0.95, profile=_profile())
    assert glossary.output == "Aang arrived."
    assert local.calls == []

    uncertain = corrector.correct("The isle returned.", confidence=0.2, profile=_profile())
    assert uncertain.output == "The Avatar returned."
    assert len(local.calls) == 1
    no_model = ConfidenceGatedCorrector(GlossaryNormalizer()).correct(
        "uncertain", confidence=0.2, profile=_profile()
    )
    assert no_model.reason == "no-local-corrector"
    with pytest.raises(ValueError):
        corrector.correct("bad", confidence=2, profile=_profile())


def test_cue_processor_changes_text_without_timing() -> None:
    processor = ProfileCueProcessor(_profile(), GlossaryNormalizer())
    assert processor.process([Cue(1, 2, "Ang.")]) == [Cue(1, 2, "Aang.")]


def test_local_fuzzy_corrector_is_confidence_gated_in_cue_pipeline() -> None:
    local = ConservativeLocalCorrector(similarity_threshold=0.8)
    gated = ConfidenceGatedCorrector(GlossaryNormalizer(), local)
    processor = ProfileCueProcessor(_profile(), GlossaryNormalizer(), gated)
    corrected = processor.process([Cue(1, 2, "Bugy. law.", confidence=0.2)])
    assert corrected == [Cue(1, 2, "Buggy. law.", confidence=0.2)]
    with pytest.raises(ValueError):
        ConservativeLocalCorrector(similarity_threshold=2)
