from pathlib import Path

import pytest
import yaml

from subtitlegen.evaluation.metrics import (
    mean_timestamp_error,
    terminology_recall,
    word_error_rate,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "avatar_asr_annotations.yaml"


def test_word_error_rate_handles_edits_and_empty_references() -> None:
    assert word_error_rate("one two three", "one too three") == pytest.approx(1 / 3)
    assert word_error_rate("", "") == 0
    assert word_error_rate("", "unexpected") == 1


def test_terminology_recall_matches_complete_case_insensitive_terms() -> None:
    assert terminology_recall(("Avatar", "airbender"), "An AIRBENDER is the avatar.") == 1
    assert terminology_recall(("Avatar", "airbender"), "avatars") == 0
    assert terminology_recall((), "anything") == 1


def test_timestamp_error_validates_shape_and_uses_annotations() -> None:
    fixture = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    reference = fixture["phrase_onsets_seconds"]
    assert mean_timestamp_error(reference, [0.82, 6.1, 10.98]) == 0
    assert mean_timestamp_error(reference, [0.0, 6.04, 10.98]) == pytest.approx(0.2933, abs=1e-4)
    assert mean_timestamp_error([], []) == 0
    with pytest.raises(ValueError):
        mean_timestamp_error([0], [])
