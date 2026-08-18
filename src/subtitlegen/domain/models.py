from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Word:
    start: float
    end: float
    text: str
    probability: float | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("word start must be non-negative")
        if self.end < self.start:
            raise ValueError("word end must not precede its start")
        if not self.text.strip():
            raise ValueError("word text must not be empty")
        if self.probability is not None and not 0 <= self.probability <= 1:
            raise ValueError("word probability must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class Cue:
    start: float
    end: float
    text: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("cue start must be non-negative")
        if self.end < self.start:
            raise ValueError("cue end must not precede its start")
        if not self.text.strip():
            raise ValueError("cue text must not be empty")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("cue confidence must be between 0 and 1")

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Transcription:
    words: tuple[Word, ...]
    language: str
    duration: float | None = None

    def __post_init__(self) -> None:
        if not self.language:
            raise ValueError("transcription language must not be empty")
        if self.duration is not None and self.duration < 0:
            raise ValueError("transcription duration must be non-negative")
        previous_start = 0.0
        for word in self.words:
            if word.start < previous_start:
                raise ValueError("transcription words must be ordered")
            previous_start = word.start
