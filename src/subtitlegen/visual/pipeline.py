from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from subtitlegen.visual.detection import TextDetector
from subtitlegen.visual.models import BoundingBox, SampledFrame, VisualEvent, VisualObservation
from subtitlegen.visual.ocr import OcrEngine, contains_japanese
from subtitlegen.visual.sampler import FrameSampler
from subtitlegen.visual.tracker import VisualEventTracker, perceptual_hash
from subtitlegen.visual.translation import Translator


class VisualTextPipeline:
    def __init__(
        self,
        sampler: FrameSampler,
        detector: TextDetector,
        ocr: OcrEngine,
        translator: Translator,
        tracker: VisualEventTracker,
        *,
        crop_padding_ratio: float = 0.0,
        minimum_box_area_ratio: float = 0.01,
        minimum_vertical_center_ratio: float = 0.45,
    ) -> None:
        if not 0 <= crop_padding_ratio <= 0.5:
            raise ValueError("crop padding ratio must be within [0, 0.5]")
        if not 0 <= minimum_box_area_ratio <= 1:
            raise ValueError("minimum box area ratio must be within [0, 1]")
        if not 0 <= minimum_vertical_center_ratio <= 1:
            raise ValueError("minimum vertical center ratio must be within [0, 1]")
        self._sampler = sampler
        self._detector = detector
        self._ocr = ocr
        self._translator = translator
        self._tracker = tracker
        self._crop_padding_ratio = crop_padding_ratio
        self._minimum_box_area_ratio = minimum_box_area_ratio
        self._minimum_vertical_center_ratio = minimum_vertical_center_ratio

    def process(self, media_path: Path) -> tuple[VisualEvent, ...]:
        observations: list[VisualObservation] = []
        cache: dict[bytes, tuple[str, str]] = {}
        batch: list[SampledFrame] = []
        for frame in self._sampler.sample(media_path):
            batch.append(frame)
            if len(batch) == 16:
                self._process_batch(batch, observations, cache)
                batch.clear()
        self._process_batch(batch, observations, cache)
        return self._tracker.track(observations)

    def _process_batch(
        self,
        frames: list[SampledFrame],
        observations: list[VisualObservation],
        cache: dict[bytes, tuple[str, str]],
    ) -> None:
        if not frames:
            return
        images = [np.asarray(frame.image) for frame in frames]
        detect_batch = getattr(self._detector, "detect_batch", None)
        detections = (
            detect_batch(images)
            if detect_batch is not None
            else tuple(self._detector.detect(image) for image in images)
        )
        for frame, image, detected in zip(frames, images, detections, strict=True):
            boxes = tuple(detected)
            for box in self._eligible_boxes(boxes, image.shape[1], image.shape[0]):
                crop = self._crop(
                    image,
                    box.x,
                    box.y,
                    box.width,
                    box.height,
                    padding_ratio=self._crop_padding_ratio,
                )
                if crop.size == 0:
                    continue
                image_hash = perceptual_hash(crop)
                fingerprint = self._fingerprint(crop)
                cached = cache.get(fingerprint)
                if cached is None:
                    result = self._ocr.recognize(crop)
                    source_text = result.text.strip()
                    if not contains_japanese(source_text):
                        continue
                    translated_text = self._translator.translate(source_text)
                    cache[fingerprint] = (source_text, translated_text)
                else:
                    source_text, translated_text = cached
                observations.append(
                    VisualObservation(
                        frame.timestamp,
                        box,
                        source_text,
                        translated_text,
                        image_hash,
                    )
                )

    def close(self) -> None:
        for component in (self._detector, self._ocr, self._translator):
            close = getattr(component, "close", None)
            if close is not None:
                close()

    @staticmethod
    def _fingerprint(image: np.ndarray[Any, Any]) -> bytes:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(image.shape).encode())
        digest.update(np.ascontiguousarray(image).tobytes())
        return digest.digest()

    @staticmethod
    def _crop(
        image: np.ndarray[Any, Any],
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        padding_ratio: float = 0.0,
    ) -> np.ndarray[Any, Any]:
        horizontal_padding = round(width * padding_ratio)
        vertical_padding = round(height * padding_ratio)
        top = max(0, y - vertical_padding)
        left = max(0, x - horizontal_padding)
        bottom = min(image.shape[0], y + height + vertical_padding)
        right = min(image.shape[1], x + width + horizontal_padding)
        return image[top:bottom, left:right]

    @staticmethod
    def _primary_boxes(boxes: tuple[BoundingBox, ...]) -> tuple[BoundingBox, ...]:
        return tuple(
            box
            for box in boxes
            if not any(
                box.area < other.area * 0.25
                and other.x <= box.x + box.width / 2 <= other.x + other.width
                and other.y <= box.y + box.height / 2 <= other.y + other.height
                for other in boxes
                if other is not box
            )
        )

    def _eligible_boxes(
        self,
        boxes: tuple[BoundingBox, ...],
        frame_width: int,
        frame_height: int,
    ) -> tuple[BoundingBox, ...]:
        minimum_area = frame_width * frame_height * self._minimum_box_area_ratio
        minimum_center = frame_height * self._minimum_vertical_center_ratio
        return tuple(
            box
            for box in self._primary_boxes(boxes)
            if box.area >= minimum_area and box.y + box.height / 2 >= minimum_center
        )
