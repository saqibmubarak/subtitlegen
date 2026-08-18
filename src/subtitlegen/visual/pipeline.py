from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from subtitlegen.visual.detection import TextDetector
from subtitlegen.visual.models import BoundingBox, SampledFrame, VisualEvent, VisualObservation
from subtitlegen.visual.ocr import OcrEngine, japanese_character_count
from subtitlegen.visual.proposals import RegionProposer
from subtitlegen.visual.sampler import FrameSource
from subtitlegen.visual.tracker import VisualEventTracker, perceptual_hash
from subtitlegen.visual.translation import Translator

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _DetectorRegion:
    frame_index: int
    region: BoundingBox
    scale: float = 1.0
    offset_x: int = 0
    offset_y: int = 0


@dataclass(frozen=True, slots=True)
class _RememberedText:
    box: BoundingBox
    source_text: str
    translated_text: str
    image_hash: int
    image_signature: tuple[int, ...]


class VisualTextPipeline:
    def __init__(
        self,
        sampler: FrameSource,
        detector: TextDetector,
        ocr: OcrEngine,
        translator: Translator,
        tracker: VisualEventTracker,
        *,
        region_proposer: RegionProposer | None = None,
        crop_padding_ratio: float = 0.0,
        minimum_box_area_ratio: float = 0.01,
        minimum_vertical_center_ratio: float = 0.45,
        detector_input_size: int = 416,
        minimum_japanese_characters: int = 5,
    ) -> None:
        if not 0 <= crop_padding_ratio <= 0.5:
            raise ValueError("crop padding ratio must be within [0, 0.5]")
        if not 0 <= minimum_box_area_ratio <= 1:
            raise ValueError("minimum box area ratio must be within [0, 1]")
        if not 0 <= minimum_vertical_center_ratio <= 1:
            raise ValueError("minimum vertical center ratio must be within [0, 1]")
        if detector_input_size < 32:
            raise ValueError("detector input size must be at least 32 pixels")
        if minimum_japanese_characters <= 0:
            raise ValueError("minimum Japanese character count must be positive")
        self._sampler = sampler
        self._detector = detector
        self._ocr = ocr
        self._translator = translator
        self._tracker = tracker
        self._region_proposer = region_proposer
        self._crop_padding_ratio = crop_padding_ratio
        self._minimum_box_area_ratio = minimum_box_area_ratio
        self._minimum_vertical_center_ratio = minimum_vertical_center_ratio
        self._detector_input_size = detector_input_size
        self._minimum_japanese_characters = minimum_japanese_characters

    def process(self, media_path: Path) -> tuple[VisualEvent, ...]:
        if self._region_proposer is not None:
            self._region_proposer.reset()
        observations: list[VisualObservation] = []
        cache: dict[bytes, tuple[str, str]] = {}
        region_memory: dict[tuple[int, int, int, int], tuple[_RememberedText, ...]] = {}
        batch: list[SampledFrame] = []
        for frame in self._sampler.sample(media_path):
            batch.append(frame)
            batch_size = 4 if self._region_proposer is not None else 16
            if len(batch) == batch_size:
                self._process_batch(batch, observations, cache, region_memory)
                batch.clear()
        self._process_batch(batch, observations, cache, region_memory)
        return self._tracker.track(observations)

    def _process_batch(
        self,
        frames: list[SampledFrame],
        observations: list[VisualObservation],
        cache: dict[bytes, tuple[str, str]],
        region_memory: dict[
            tuple[int, int, int, int],
            tuple[_RememberedText, ...],
        ],
    ) -> None:
        if not frames:
            return
        images = [np.asarray(frame.image) for frame in frames]
        regions = [
            (
                tuple(
                    self._region_proposer.propose(
                        image,
                        scene_change=frame.scene_change,
                    )
                )
                if self._region_proposer is not None
                else (BoundingBox(0, 0, image.shape[1], image.shape[0]),)
            )
            for frame, image in zip(frames, images, strict=True)
        ]
        remembered_regions = tuple(
            BoundingBox(*region_key)
            for region_key in region_memory
        )
        if remembered_regions:
            regions = [
                proposed or remembered_regions
                for proposed in regions
            ]
        detector_inputs: list[np.ndarray[Any, Any]] = []
        detector_mapping: list[_DetectorRegion] = []
        for frame_index, (image, proposed) in enumerate(
            zip(images, regions, strict=True)
        ):
            for region in proposed:
                region_key = self._region_key(region)
                remembered = region_memory.get(region_key)
                if remembered is not None:
                    refreshed = self._refresh_remembered(
                        frames[frame_index],
                        image,
                        remembered,
                    )
                    if refreshed is not None:
                        observations.extend(refreshed)
                        continue
                    region_memory.pop(region_key, None)
                crop = self._crop(
                    image,
                    region.x,
                    region.y,
                    region.width,
                    region.height,
                )
                if crop.size:
                    if self._region_proposer is None:
                        detector_inputs.append(crop)
                        detector_mapping.append(_DetectorRegion(frame_index, region))
                    else:
                        prepared, scale, offset_x, offset_y = (
                            self._prepare_detector_input(crop)
                        )
                        detector_inputs.append(prepared)
                        detector_mapping.append(
                            _DetectorRegion(
                                frame_index,
                                region,
                                scale,
                                offset_x,
                                offset_y,
                            )
                        )
        if not detector_inputs:
            return
        local_detections = self._detect_inputs(detector_inputs)
        detections: list[list[tuple[BoundingBox, BoundingBox]]] = [[] for _ in frames]
        for mapping, local_boxes in zip(
            detector_mapping,
            local_detections,
            strict=True,
        ):
            frame_index = mapping.frame_index
            region = mapping.region
            frame_height, frame_width = images[frame_index].shape[:2]
            for box in local_boxes:
                local_left = max(0, round((box.x - mapping.offset_x) / mapping.scale))
                local_top = max(0, round((box.y - mapping.offset_y) / mapping.scale))
                local_right = min(
                    region.width,
                    round(
                        (box.x + box.width - mapping.offset_x)
                        / mapping.scale
                    ),
                )
                local_bottom = min(
                    region.height,
                    round(
                        (box.y + box.height - mapping.offset_y)
                        / mapping.scale
                    ),
                )
                x = region.x + local_left
                y = region.y + local_top
                width = min(local_right - local_left, frame_width - x)
                height = min(local_bottom - local_top, frame_height - y)
                if width > 0 and height > 0:
                    detections[frame_index].append(
                        (BoundingBox(x, y, width, height, box.score), region)
                    )
        for frame, image, frame_detections in zip(
            frames,
            images,
            detections,
            strict=True,
        ):
            boxes = self._deduplicate_boxes(
                tuple(box for box, _ in frame_detections)
            )
            remembered_by_region: dict[
                tuple[int, int, int, int],
                list[_RememberedText],
            ] = {}
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
                    if (
                        japanese_character_count(source_text)
                        < self._minimum_japanese_characters
                    ):
                        logger.debug(
                            "filtered short visual OCR candidate at %.3fs: %r",
                            frame.timestamp,
                            source_text,
                        )
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
                region = next(
                    region
                    for detected_box, region in frame_detections
                    if detected_box == box
                )
                remembered_by_region.setdefault(self._region_key(region), []).append(
                    _RememberedText(
                        box,
                        source_text,
                        translated_text,
                        image_hash,
                        self._visual_signature(crop),
                    )
                )
            region_memory.update(
                {
                    key: tuple(remembered)
                    for key, remembered in remembered_by_region.items()
                }
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

    def _prepare_detector_input(
        self,
        image: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], float, int, int]:
        height, width = image.shape[:2]
        scale = min(
            self._detector_input_size / width,
            self._detector_input_size / height,
        )
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        x_indices = np.linspace(0, width - 1, resized_width).astype(int)
        y_indices = np.linspace(0, height - 1, resized_height).astype(int)
        resized = image[np.ix_(y_indices, x_indices)]
        offset_x = (self._detector_input_size - resized_width) // 2
        offset_y = (self._detector_input_size - resized_height) // 2
        output_shape = (
            self._detector_input_size,
            self._detector_input_size,
            *image.shape[2:],
        )
        prepared = np.full(output_shape, round(float(np.mean(image))), dtype=image.dtype)
        prepared[
            offset_y : offset_y + resized_height,
            offset_x : offset_x + resized_width,
        ] = resized
        return prepared, scale, offset_x, offset_y

    def _refresh_remembered(
        self,
        frame: SampledFrame,
        image: np.ndarray[Any, Any],
        remembered: tuple[_RememberedText, ...],
    ) -> tuple[VisualObservation, ...] | None:
        refreshed: list[VisualObservation] = []
        for item in remembered:
            crop = self._crop(
                image,
                item.box.x,
                item.box.y,
                item.box.width,
                item.box.height,
                padding_ratio=self._crop_padding_ratio,
            )
            if crop.size == 0:
                return None
            image_hash = perceptual_hash(crop)
            if (image_hash ^ item.image_hash).bit_count() > 8:
                return None
            signature = self._visual_signature(crop)
            mean_difference = sum(
                abs(current - previous)
                for current, previous in zip(
                    signature,
                    item.image_signature,
                    strict=True,
                )
            ) / len(signature)
            if mean_difference > 12:
                return None
            refreshed.append(
                VisualObservation(
                    frame.timestamp,
                    item.box,
                    item.source_text,
                    item.translated_text,
                    image_hash,
                )
            )
        return tuple(refreshed)

    @staticmethod
    def _region_key(region: BoundingBox) -> tuple[int, int, int, int]:
        return region.x, region.y, region.width, region.height

    @staticmethod
    def _visual_signature(image: np.ndarray[Any, Any]) -> tuple[int, ...]:
        gray = (
            image.astype(np.float32).mean(axis=2)
            if image.ndim == 3
            else image.astype(np.float32)
        )
        y_indices = np.linspace(0, gray.shape[0] - 1, 16).astype(int)
        x_indices = np.linspace(0, gray.shape[1] - 1, 16).astype(int)
        return tuple(int(value) for value in gray[np.ix_(y_indices, x_indices)].flat)

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

    @staticmethod
    def _deduplicate_boxes(boxes: tuple[BoundingBox, ...]) -> tuple[BoundingBox, ...]:
        selected: list[BoundingBox] = []
        for box in sorted(boxes, key=lambda item: item.score, reverse=True):
            if all(box.intersection_over_union(other) < 0.8 for other in selected):
                selected.append(box)
        return tuple(selected)

    def _detect_inputs(
        self,
        images: list[np.ndarray[Any, Any]],
    ) -> tuple[tuple[BoundingBox, ...], ...]:
        detect_batch = getattr(self._detector, "detect_batch", None)
        if detect_batch is None:
            return tuple(tuple(self._detector.detect(image)) for image in images)
        grouped: dict[tuple[int, ...], list[int]] = {}
        for index, image in enumerate(images):
            grouped.setdefault(tuple(image.shape), []).append(index)
        results: list[tuple[BoundingBox, ...] | None] = [None] * len(images)
        for indices in grouped.values():
            detected = detect_batch([images[index] for index in indices])
            for index, boxes in zip(indices, detected, strict=True):
                results[index] = tuple(boxes)
        if any(result is None for result in results):
            raise RuntimeError("text detector omitted a proposed region")
        return tuple(result for result in results if result is not None)
