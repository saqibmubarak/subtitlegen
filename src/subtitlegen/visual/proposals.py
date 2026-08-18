from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from subtitlegen.errors import BackendUnavailableError
from subtitlegen.visual.models import BoundingBox


class RegionProposer(Protocol):
    def propose(self, image: Any, *, scene_change: bool = False) -> Sequence[BoundingBox]:
        """Return inexpensive regions that should be sent to the text detector."""

    def reset(self) -> None:
        """Clear temporal state before processing another video."""


ComponentExtractor = Callable[
    [np.ndarray[Any, Any]],
    Sequence[tuple[int, int, int, int, int]],
]


class TemporalDifferenceProposer:
    """Find changing screen regions cheaply before model-based text detection."""

    def __init__(
        self,
        *,
        difference_threshold: int = 24,
        minimum_changed_area_ratio: float = 0.001,
        maximum_changed_area_ratio: float = 0.35,
        padding_ratio: float = 0.5,
        hold_frames: int = 12,
        full_frame_hold_frames: int = 4,
        analysis_width: int = 320,
        maximum_regions: int = 2,
        component_extractor: ComponentExtractor | None = None,
    ) -> None:
        if not 0 < difference_threshold <= 255:
            raise ValueError("difference threshold must be within [1, 255]")
        if not 0 < minimum_changed_area_ratio < maximum_changed_area_ratio <= 1:
            raise ValueError("changed area ratios are invalid")
        if not 0 <= padding_ratio <= 0.5:
            raise ValueError("proposal padding ratio must be within [0, 0.5]")
        if (
            hold_frames <= 0
            or full_frame_hold_frames <= 0
            or analysis_width < 32
            or maximum_regions <= 0
        ):
            raise ValueError("proposal sizing settings must be positive")
        self._difference_threshold = difference_threshold
        self._minimum_area_ratio = minimum_changed_area_ratio
        self._maximum_area_ratio = maximum_changed_area_ratio
        self._padding_ratio = padding_ratio
        self._hold_frames = hold_frames
        self._full_frame_hold_frames = full_frame_hold_frames
        self._analysis_width = analysis_width
        self._maximum_regions = maximum_regions
        self._component_extractor = component_extractor
        self._previous: np.ndarray[Any, Any] | None = None
        self._held: list[tuple[BoundingBox, int]] = []

    def reset(self) -> None:
        self._previous = None
        self._held.clear()

    def propose(
        self,
        image: Any,
        *,
        scene_change: bool = False,
    ) -> tuple[BoundingBox, ...]:
        array = np.asarray(image)
        if array.ndim not in (2, 3) or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("proposal image must be a non-empty grayscale or RGB array")
        frame_height, frame_width = array.shape[:2]
        gray = self._analysis_image(array)
        previous = self._previous
        self._previous = gray
        held = [(box, ttl - 1) for box, ttl in self._held if ttl > 1]

        if previous is None or previous.shape != gray.shape:
            full = BoundingBox(0, 0, frame_width, frame_height)
            self._held = [(full, self._full_frame_hold_frames)]
            return (full,)
        held_full = next(
            (
                box
                for box, _ in held
                if box.x == 0
                and box.y == 0
                and box.width == frame_width
                and box.height == frame_height
            ),
            None,
        )
        difference = np.abs(gray.astype(np.int16) - previous.astype(np.int16))
        mask = (difference >= self._difference_threshold).astype(np.uint8)
        changed_ratio = float(mask.mean())
        if changed_ratio >= self._maximum_area_ratio:
            full = BoundingBox(0, 0, frame_width, frame_height)
            self._held = [(full, self._full_frame_hold_frames)]
            return (full,)

        proposals = self._regions(mask, frame_width, frame_height)
        if held_full is not None and not proposals:
            self._held = held
            return (held_full,)
        if scene_change and not proposals:
            full = BoundingBox(0, 0, frame_width, frame_height)
            self._held = [(full, self._full_frame_hold_frames)]
            return (full,)
        if held_full is not None:
            held = [
                (box, ttl)
                for box, ttl in held
                if box != held_full
            ]
        held.extend((box, self._hold_frames) for box in proposals)
        self._held = self._merge_held(held)
        return tuple(box for box, _ in self._held)

    def _analysis_image(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        gray = (
            image.astype(np.float32).mean(axis=2)
            if image.ndim == 3
            else image
        )
        height = max(18, round(self._analysis_width * gray.shape[0] / gray.shape[1]))
        y_indices = np.linspace(0, gray.shape[0] - 1, height).astype(int)
        x_indices = np.linspace(0, gray.shape[1] - 1, self._analysis_width).astype(int)
        return np.asarray(gray[np.ix_(y_indices, x_indices)], dtype=np.uint8)

    def _regions(
        self,
        mask: np.ndarray[Any, Any],
        frame_width: int,
        frame_height: int,
    ) -> tuple[BoundingBox, ...]:
        extractor = self._component_extractor or self._opencv_components
        components = extractor(mask)
        minimum_area = mask.shape[0] * mask.shape[1] * self._minimum_area_ratio
        scale_x = frame_width / mask.shape[1]
        scale_y = frame_height / mask.shape[0]
        boxes: list[BoundingBox] = []
        for x, y, width, height, area in sorted(
            components,
            key=lambda item: item[4],
            reverse=True,
        ):
            if area < minimum_area:
                continue
            left = round(x * scale_x)
            top = round(y * scale_y)
            right = round((x + width) * scale_x)
            bottom = round((y + height) * scale_y)
            padding_x = round((right - left) * self._padding_ratio)
            padding_y = round((bottom - top) * self._padding_ratio)
            left = max(0, left - padding_x)
            top = max(0, top - padding_y)
            right = min(frame_width, right + padding_x)
            bottom = min(frame_height, bottom + padding_y)
            if right > left and bottom > top:
                boxes.append(BoundingBox(left, top, right - left, bottom - top))
            if len(boxes) == self._maximum_regions:
                break
        return tuple(boxes)

    @staticmethod
    def _opencv_components(
        mask: np.ndarray[Any, Any],
    ) -> Sequence[tuple[int, int, int, int, int]]:
        try:
            import cv2
        except ImportError as error:
            raise BackendUnavailableError(
                "temporal region proposals require subtitlegen[ocr]"
            ) from error
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        connected = cv2.dilate(connected, kernel, iterations=1)
        count, _, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
        return tuple(
            (
                int(stats[index, cv2.CC_STAT_LEFT]),
                int(stats[index, cv2.CC_STAT_TOP]),
                int(stats[index, cv2.CC_STAT_WIDTH]),
                int(stats[index, cv2.CC_STAT_HEIGHT]),
                int(stats[index, cv2.CC_STAT_AREA]),
            )
            for index in range(1, count)
        )

    def _merge_held(
        self,
        held: list[tuple[BoundingBox, int]],
    ) -> list[tuple[BoundingBox, int]]:
        merged: list[tuple[BoundingBox, int]] = []
        for box, ttl in held:
            match = next(
                (
                    index
                    for index, (existing, _) in enumerate(merged)
                    if existing.intersection_over_union(box) >= 0.5
                ),
                None,
            )
            if match is None:
                merged.append((box, ttl))
            else:
                existing, existing_ttl = merged[match]
                merged[match] = (existing, max(existing_ttl, ttl))
        return sorted(
            merged,
            key=lambda item: (item[0].y + item[0].height / 2, item[0].area),
            reverse=True,
        )[: self._maximum_regions]
