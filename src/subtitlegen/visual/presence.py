from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from subtitlegen.visual.detection import TextDetector
from subtitlegen.visual.models import BoundingBox
from subtitlegen.visual.ocr import OcrEngine, contains_japanese, rotate_vertical_crop


@dataclass(frozen=True, slots=True)
class PresenceDecision:
    accepted: bool
    reason: str
    box_count: int
    recognized: tuple[str, ...]
    skipped_crops: int = 0
    orientations: tuple[str, ...] = ()
    orientations: tuple[str, ...] = ()


class JapaneseCharacterScanner:
    """Downscale a frame, detect text, and accept it if any crop is Japanese."""

    def __init__(
        self,
        detector: TextDetector,
        recognizer: OcrEngine,
        *,
        analysis_width: int = 480,
        maximum_crops: int = 16,
        downscale: Callable[[Any, int], Any] | None = None,
    ) -> None:
        if analysis_width < 32:
            raise ValueError("analysis width must be at least 32 pixels")
        if maximum_crops <= 0:
            raise ValueError("maximum crops must be positive")
        self._detector = detector
        self._recognizer = recognizer
        self._analysis_width = analysis_width
        self._maximum_crops = maximum_crops
        self._downscale = downscale

    def contains_japanese(self, image: Any) -> bool:
        return self.inspect(image).accepted

    def inspect(self, image: Any) -> PresenceDecision:
        array = np.asarray(image)
        if array.size == 0:
            return PresenceDecision(False, "empty_frame", 0, ())
        prepared, scale = self._prepare(array)
        boxes = tuple(self._detector.detect(prepared))
        if not boxes:
            return PresenceDecision(False, "no_boxes", 0, ())
        ranked = sorted(boxes, key=lambda item: item.area, reverse=True)
        inspected = ranked[: self._maximum_crops]
        recognized: list[str] = []
        orientations: list[str] = []
        for box in inspected:
            crop = self._crop(array, box, scale)
            if crop.size == 0:
                recognized.append("")
                orientations.append("empty")
                continue
            text, orientation = self._recognize(crop, box)
            recognized.append(text)
            orientations.append(orientation)
            if contains_japanese(text):
                return PresenceDecision(
                    True,
                    "hit",
                    len(boxes),
                    tuple(recognized),
                    skipped_crops=max(0, len(ranked) - len(inspected)),
                    orientations=tuple(orientations),
                )
        return PresenceDecision(
            False,
            "no_japanese",
            len(boxes),
            tuple(recognized),
            skipped_crops=max(0, len(ranked) - len(inspected)),
            orientations=tuple(orientations),
        )

    def _recognize(self, crop: np.ndarray[Any, Any], box: BoundingBox) -> tuple[str, str]:
        text = self._recognizer.recognize(crop).text.strip()
        if contains_japanese(text):
            return text, "vertical" if box.is_vertical() else "horizontal"
        if not box.is_vertical():
            return text, "horizontal"
        rotated = rotate_vertical_crop(crop)
        rotated_text = self._recognizer.recognize(rotated).text.strip()
        if contains_japanese(rotated_text):
            return rotated_text, "vertical-rotated"
        return text, "vertical"

    def close(self) -> None:
        for component in (self._recognizer,):
            close = getattr(component, "close", None)
            if close is not None:
                close()

    def _prepare(self, image: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], float]:
        height, width = image.shape[:2]
        if width <= self._analysis_width:
            return image, 1.0
        if self._downscale is not None:
            return self._downscale(image, self._analysis_width), width / self._analysis_width
        scale = self._analysis_width / width
        resized_height = max(1, round(height * scale))
        x_indices = np.linspace(0, width - 1, self._analysis_width).astype(int)
        y_indices = np.linspace(0, height - 1, resized_height).astype(int)
        return image[np.ix_(y_indices, x_indices)], 1 / scale

    @staticmethod
    def _crop(
        image: np.ndarray[Any, Any],
        box: BoundingBox,
        scale: float,
    ) -> np.ndarray[Any, Any]:
        x = max(0, round(box.x * scale))
        y = max(0, round(box.y * scale))
        width = max(1, round(box.width * scale))
        height = max(1, round(box.height * scale))
        return image[y : y + height, x : x + width]
