import pytest

from subtitlegen.domain.models import Cue, Transcription, Word


def test_word_validates_boundaries_and_probability() -> None:
    word = Word(0.0, 0.5, "Hello", 0.9)
    assert word.text == "Hello"
    with pytest.raises(ValueError):
        Word(-1, 0, "bad")
    with pytest.raises(ValueError):
        Word(1, 0, "bad")
    with pytest.raises(ValueError):
        Word(0, 1, " ")
    with pytest.raises(ValueError):
        Word(0, 1, "bad", 2)


def test_cue_exposes_duration_and_validates() -> None:
    cue = Cue(1.0, 3.5, "Hello")
    assert cue.duration == 2.5
    with pytest.raises(ValueError):
        Cue(-1, 1, "bad")
    with pytest.raises(ValueError):
        Cue(2, 1, "bad")
    with pytest.raises(ValueError):
        Cue(0, 1, "bad", confidence=2)


def test_transcription_requires_ordered_words_and_language() -> None:
    words = (Word(0, 1, "one"), Word(1, 2, "two"))
    assert Transcription(words, "en", 2).words == words
    with pytest.raises(ValueError):
        Transcription(tuple(reversed(words)), "en")
    with pytest.raises(ValueError):
        Transcription(words, "")
