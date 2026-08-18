from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from subtitlegen.cues.builder import CueBuilder
from subtitlegen.cues.rules import CueRules
from subtitlegen.domain.models import Word


def test_cue_rules_validate_values() -> None:
    assert CueRules().max_duration_seconds == 6
    with pytest.raises(ValueError):
        CueRules(max_duration_seconds=0)
    with pytest.raises(ValueError):
        CueRules(max_characters=0)
    with pytest.raises(ValueError):
        CueRules(max_gap_seconds=-1)


def test_builder_flushes_on_gap_duration_characters_and_punctuation() -> None:
    words = [
        Word(0.0, 0.4, "Hello"),
        Word(0.4, 1.1, " world."),
        Word(2.0, 2.5, "A"),
        Word(2.5, 3.0, " very"),
        Word(3.0, 3.5, " long"),
        Word(3.5, 4.0, " phrase"),
    ]
    cues = CueBuilder(CueRules(max_characters=12)).build(words)
    assert [cue.text for cue in cues] == ["Hello world.", "A very long", "phrase"]
    assert all(cue.duration <= 6 for cue in cues)


def test_builder_normalizes_overlapping_word_ranges() -> None:
    cues = CueBuilder().build([Word(0, 2, "one."), Word(1.9, 3, "two.")])
    assert cues[1].start == cues[0].end
    assert cues[1].end >= cues[1].start


def test_builder_drops_cues_fully_covered_by_previous_timing() -> None:
    cues = CueBuilder().build([Word(0, 3, "one."), Word(1, 2, "duplicate.")])
    assert [cue.text for cue in cues] == ["one."]


def test_builder_drops_a_single_oversized_hallucinated_word() -> None:
    cues = CueBuilder(CueRules(max_duration_seconds=6)).build([Word(0, 9, "drawn-out")])
    assert cues == []


def test_builder_uses_lowest_word_confidence_for_correction_gate() -> None:
    cue = CueBuilder().build(
        [
            Word(0, 1, "Clear", probability=0.99),
            Word(1, 2, " Bugy.", probability=0.2),
        ]
    )[0]
    assert cue.confidence == 0.2


@given(
    durations=st.lists(
        st.floats(min_value=0.05, max_value=0.8, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=30,
    )
)
def test_builder_always_returns_ordered_non_overlapping_cues(
    durations: list[float],
) -> None:
    words: list[Word] = []
    position = 0.0
    for index, duration in enumerate(durations):
        words.append(Word(position, position + duration, f"w{index}"))
        position += duration + 0.05

    cues = CueBuilder().build(words)
    assert all(cue.start >= 0 and cue.end >= cue.start for cue in cues)
    assert all(left.end <= right.start for left, right in pairwise(cues))
