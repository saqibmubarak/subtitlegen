from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    score: float = 1.0

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("bounding box coordinates and dimensions are invalid")
        if not 0 <= self.score <= 1:
            raise ValueError("bounding box score must be between zero and one")

    @property
    def area(self) -> int:
        return self.width * self.height

    def is_vertical(self, *, ratio: float = 1.4) -> bool:
        """True for tall crops typical of tate-gaki (top-to-bottom) Japanese."""
        return self.height >= self.width * ratio

    def intersection_over_union(self, other: BoundingBox) -> float:
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.x + self.width, other.x + other.width)
        bottom = min(self.y + self.height, other.y + other.height)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0


@dataclass(frozen=True, slots=True)
class SampledFrame:
    timestamp: float
    image: Any
    scene_change: bool = False

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("frame timestamp must be non-negative")


@dataclass(frozen=True, slots=True)
class OcrResult:
    text: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("OCR confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class VisualObservation:
    timestamp: float
    box: BoundingBox
    source_text: str
    translated_text: str
    image_hash: int

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("observation timestamp must be non-negative")
        if not self.source_text.strip() or not self.translated_text.strip():
            raise ValueError("visual observation text must not be blank")


@dataclass(frozen=True, slots=True)
class VisualEvent:
    start: float
    end: float
    source_text: str
    translated_text: str
    box: BoundingBox
    category: str = "OnScreen"

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("visual event interval is invalid")
        if not self.translated_text.strip():
            raise ValueError("translated visual text must not be blank")


@dataclass(frozen=True, slots=True)
class StyledCue:
    start: float
    end: float
    text: str
    style: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("styled cue interval is invalid")
        if not self.text.strip() or not self.style.strip():
            raise ValueError("styled cue text and style must not be blank")
