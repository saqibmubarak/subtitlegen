from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import numpy as np

from subtitlegen.visual.models import VisualEvent, VisualObservation


def perceptual_hash(image: Any) -> int:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = array.mean(axis=2)
    if array.size == 0:
        raise ValueError("cannot hash an empty image")
    y_indices = np.linspace(0, array.shape[0] - 1, 8).astype(int)
    x_indices = np.linspace(0, array.shape[1] - 1, 9).astype(int)
    sample = array[np.ix_(y_indices, x_indices)]
    bits = sample[:, 1:] >= sample[:, :-1]
    result = 0
    for bit in bits.flat:
        result = (result << 1) | int(bit)
    return result


class VisualEventTracker:
    def __init__(
        self,
        *,
        max_gap_seconds: float = 1.25,
        frame_interval_seconds: float = 2 / 3,
        min_observations: int = 2,
        box_iou_threshold: float = 0.25,
        text_similarity_threshold: float = 0.65,
        hash_distance_threshold: int = 8,
    ) -> None:
        if max_gap_seconds <= 0 or frame_interval_seconds <= 0:
            raise ValueError("tracker timing settings must be positive")
        if min_observations <= 0:
            raise ValueError("minimum observations must be positive")
        if not 0 <= box_iou_threshold <= 1 or not 0 <= text_similarity_threshold <= 1:
            raise ValueError("tracker similarity thresholds must be within [0, 1]")
        if not 0 <= hash_distance_threshold <= 64:
            raise ValueError("hash distance threshold must be within [0, 64]")
        self._max_gap = max_gap_seconds
        self._frame_interval = frame_interval_seconds
        self._min_observations = min_observations
        self._box_iou = box_iou_threshold
        self._text_similarity = text_similarity_threshold
        self._hash_distance = hash_distance_threshold

    def track(self, observations: list[VisualObservation]) -> tuple[VisualEvent, ...]:
        tracks: list[list[VisualObservation]] = []
        for observation in sorted(observations, key=lambda item: item.timestamp):
            candidates = [
                track
                for track in tracks
                if observation.timestamp - track[-1].timestamp <= self._max_gap
                and self._matches(track[-1], observation)
            ]
            if candidates:
                best = max(
                    candidates,
                    key=lambda track: track[-1].box.intersection_over_union(observation.box),
                )
                best.append(observation)
            else:
                tracks.append([observation])

        events = [
            VisualEvent(
                start=track[0].timestamp,
                end=track[-1].timestamp + self._frame_interval,
                source_text=track[-1].source_text,
                translated_text=track[-1].translated_text,
                box=track[-1].box,
            )
            for track in tracks
            if len(track) >= self._min_observations
        ]
        return tuple(sorted(events, key=lambda event: (event.start, event.end)))

    def _matches(self, previous: VisualObservation, current: VisualObservation) -> bool:
        if previous.box.intersection_over_union(current.box) < self._box_iou:
            return False
        text_similarity = SequenceMatcher(
            None,
            previous.source_text.casefold(),
            current.source_text.casefold(),
        ).ratio()
        hash_distance = (previous.image_hash ^ current.image_hash).bit_count()
        return (
            text_similarity >= self._text_similarity
            and (
                hash_distance <= self._hash_distance
                or text_similarity >= 0.85
            )
        )
