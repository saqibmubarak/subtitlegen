import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import av
import pytest
import yaml

from subtitlegen.evaluation.metrics import character_error_rate
from subtitlegen.profiles.repository import ProfileRepository
from subtitlegen.visual.detection import PaddleOcrDetector
from subtitlegen.visual.models import BoundingBox
from subtitlegen.visual.ocr import MangaOcrEngine
from subtitlegen.visual.pipeline import VisualTextPipeline
from subtitlegen.visual.sampler import FrameSampler
from subtitlegen.visual.tracker import VisualEventTracker
from subtitlegen.visual.translation import NllbLocalTranslator

SAMPLE = Path("samples/[Muhn Pace] Dressrosa 03.mp4")
ANNOTATIONS = Path("tests/fixtures/dressrosa_visual_annotations.yaml")


def _frame_at(path: Path, timestamp: float) -> Any:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(round(timestamp * 1_000_000), backward=True)
        return next(
            frame.to_ndarray(format="rgb24")
            for frame in container.decode(stream)
            if frame.pts is not None
            and float(frame.pts * stream.time_base) >= timestamp - 0.05
        )


def _reference_annotations() -> list[tuple[Path, dict[str, Any]]]:
    data = yaml.safe_load(ANNOTATIONS.read_text(encoding="utf-8"))
    return [
        (Path("samples") / sample["video"], annotation)
        for sample in data["samples"]
        for annotation in sample["annotations"]
    ]


def _reviewed_negative_frames(_path: Path) -> Iterable[tuple[float, Any]]:
    data = yaml.safe_load(ANNOTATIONS.read_text(encoding="utf-8"))
    for sample in data["samples"]:
        path = Path("samples") / sample["video"]
        for start, end in sample.get("reviewed_negative_intervals", ()):
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                container.seek(round(start * 1_000_000), backward=True)
                next_sample = start
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    timestamp = float(frame.pts * stream.time_base)
                    if timestamp < next_sample:
                        continue
                    if timestamp > end:
                        break
                    yield timestamp, frame.to_ndarray(format="rgb24")
                    next_sample += 0.5


def _window_frames(_path: Path) -> Iterable[tuple[float, Any]]:
    with av.open(str(SAMPLE)) as container:
        stream = container.streams.video[0]
        container.seek(1_317_000_000, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            if timestamp < 1317:
                continue
            if timestamp > 1323:
                break
            yield timestamp, frame.to_ndarray(format="rgb24")


@pytest.mark.model
@pytest.mark.skipif(
    os.environ.get("SUBTITLEGEN_RUN_MODEL_TESTS") != "1" or not SAMPLE.is_file(),
    reason="requires local Dressrosa media and downloaded OCR models",
)
def test_dressrosa_scene_card_end_to_end() -> None:
    profile = ProfileRepository.default().load("one-piece")
    pipeline = VisualTextPipeline(
        FrameSampler(frame_reader=_window_frames),
        PaddleOcrDetector(),
        MangaOcrEngine(),
        NllbLocalTranslator(profile=profile),
        VisualEventTracker(),
    )

    try:
        events = pipeline.process(SAMPLE)
    finally:
        pipeline.close()
    card = next(event for event in events if "錦えもん" in event.source_text)

    assert character_error_rate("一人はぐれた錦えもん", card.source_text) <= 0.1
    assert card.translated_text == "Kin'emon Gets Separated"
    reference_start, reference_end = 1318.625, 1322.0
    intersection = max(0.0, min(card.end, reference_end) - max(card.start, reference_start))
    union = max(card.end, reference_end) - min(card.start, reference_start)
    assert intersection / union >= 0.8


@pytest.mark.model
@pytest.mark.skipif(
    os.environ.get("SUBTITLEGEN_RUN_MODEL_TESTS") != "1"
    or not all(path.is_file() for path, _ in _reference_annotations()),
    reason="requires all local Dressrosa media and downloaded OCR models",
)
def test_dressrosa_annotated_card_quality_by_category() -> None:
    references = _reference_annotations()
    images = [
        _frame_at(path, annotation["representative_timestamp"])
        for path, annotation in references
    ]
    detector = PaddleOcrDetector()
    detected = detector.detect_batch(images)
    ocr = MangaOcrEngine()
    translator = NllbLocalTranslator(
        profile=ProfileRepository.default().load("one-piece")
    )
    detected_count = 0
    canonical_count = 0
    errors: dict[str, list[float]] = {}

    for (_, annotation), image, boxes in zip(references, images, detected, strict=True):
        x, y, width, height = annotation["box"]
        reference_box = BoundingBox(x, y, width, height)
        if not boxes:
            continue
        match = max(boxes, key=reference_box.intersection_over_union)
        if reference_box.intersection_over_union(match) < 0.3:
            continue
        detected_count += 1
        crop = VisualTextPipeline._crop(
            image,
            match.x,
            match.y,
            match.width,
            match.height,
            padding_ratio=annotation.get("crop_padding_ratio", 0.0),
        )
        source = ocr.recognize(crop).text
        errors.setdefault(annotation["category"], []).append(
            character_error_rate(annotation["source_text"], source)
        )
        canonical_count += translator.translate(source) == annotation["translation"]

    assert detected_count / len(references) >= 0.9
    assert all(sum(values) / len(values) <= 0.1 for values in errors.values())
    assert canonical_count / len(references) >= 0.95
    negative_pipeline = VisualTextPipeline(
        FrameSampler(frame_reader=_reviewed_negative_frames),
        detector,
        ocr,
        translator,
        VisualEventTracker(frame_interval_seconds=0.5),
    )
    try:
        assert negative_pipeline.process(SAMPLE) == ()
    finally:
        negative_pipeline.close()
