from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CueRules:
    max_duration_seconds: float = 6.0
    max_characters: int = 84
    max_gap_seconds: float = 0.9
    punctuation_flush_min_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.max_duration_seconds <= 0:
            raise ValueError("maximum cue duration must be positive")
        if self.max_characters <= 0:
            raise ValueError("maximum cue characters must be positive")
        if self.max_gap_seconds < 0:
            raise ValueError("maximum word gap must be non-negative")
        if self.punctuation_flush_min_seconds < 0:
            raise ValueError("punctuation minimum duration must be non-negative")
