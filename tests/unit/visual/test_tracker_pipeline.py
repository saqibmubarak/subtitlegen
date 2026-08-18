from pathlib import Path
from typing import Any

import numpy as np
import pytest

from subtitlegen.visual.models import (
    BoundingBox,
    OcrResult,
    SampledFrame,
    VisualObservation,
)
from subtitlegen.visual.pipeline import VisualTextPipeline
from subtitlegen.visual.proposals import TemporalDifferenceProposer
from subtitlegen.visual.tracker import VisualEventTracker, perceptual_hash


def _observation(
    timestamp: float,
    *,
    text: str = "日本",
    image_hash: int = 1,
    x: int = 0,
) -> VisualObservation:
    return VisualObservation(
        timestamp,
        BoundingBox(x, 0, 10, 10),
        text,
        "Japan",
        image_hash,
    )


def test_tracker_handles_persistence_gaps_deduplication_and_movement() -> None:
    tracker = VisualEventTracker(max_gap_seconds=1, frame_interval_seconds=0.5)
    events = tracker.track(
        [
            _observation(0),
            _observation(0.5, text="日本。", image_hash=3),
            _observation(2),
            _observation(2.5),
            _observation(3, x=30),
        ]
    )

    assert [(event.start, event.end) for event in events] == [(0, 1), (2, 3)]
    assert events[0].source_text == "日本。"
    with pytest.raises(ValueError):
        VisualEventTracker(max_gap_seconds=0)
    with pytest.raises(ValueError):
        VisualEventTracker(min_observations=0)
    with pytest.raises(ValueError):
        VisualEventTracker(hash_distance_threshold=65)


def test_tracker_does_not_merge_different_text_with_identical_hashes() -> None:
    events = VisualEventTracker(frame_interval_seconds=0.5).track(
        [
            _observation(0, text="ドレスローザ", image_hash=1),
            _observation(0.5, text="コリーダコロシアム", image_hash=1),
        ]
    )

    assert events == ()


class FakeSampler:
    def __init__(self, image: Any) -> None:
        self.image = image

    def sample(self, _path: Path) -> tuple[SampledFrame, ...]:
        return (SampledFrame(0, self.image), SampledFrame(0.5, self.image))


class SequenceSampler:
    def __init__(self, images: tuple[Any, ...]) -> None:
        self.images = images

    def sample(self, _path: Path) -> tuple[SampledFrame, ...]:
        return tuple(
            SampledFrame(index * 0.5, image)
            for index, image in enumerate(self.images)
        )


class FakeDetector:
    def __init__(self, boxes: tuple[BoundingBox, ...] | None = None) -> None:
        self.boxes = boxes or (BoundingBox(0, 0, 8, 8),)

    def detect(self, _image: Any) -> tuple[BoundingBox, ...]:
        return self.boxes


class RecordingDetector(FakeDetector):
    def __init__(self) -> None:
        super().__init__()
        self.shapes: list[tuple[int, ...]] = []

    def detect(self, image: Any) -> tuple[BoundingBox, ...]:
        self.shapes.append(tuple(image.shape))
        return (BoundingBox(0, 0, image.shape[1], image.shape[0]),)


class BrightDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image: Any) -> tuple[BoundingBox, ...]:
        self.calls += 1
        if not np.asarray(image).any():
            return ()
        return (BoundingBox(0, 0, image.shape[1], image.shape[0]),)


class FakeRegionProposer:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def propose(
        self,
        _image: Any,
        *,
        scene_change: bool = False,
    ) -> tuple[BoundingBox, ...]:
        return (BoundingBox(2, 2, 4, 4),)


class FakeOcr:
    def __init__(self, text: str = "日本語字幕") -> None:
        self.text = text
        self.calls = 0

    def recognize(self, _image: Any) -> OcrResult:
        self.calls += 1
        return OcrResult(self.text)

    def close(self) -> None:
        self.text = ""


class ShapeRecordingOcr(FakeOcr):
    def __init__(self) -> None:
        super().__init__()
        self.shapes: list[tuple[int, ...]] = []

    def recognize(self, image: Any) -> OcrResult:
        self.shapes.append(tuple(image.shape))
        return super().recognize(image)


class FakeTranslator:
    def __init__(self) -> None:
        self.calls = 0

    def translate(self, _text: str) -> str:
        self.calls += 1
        return "Japan"

    def close(self) -> None:
        self.calls = -1


def test_visual_pipeline_detects_before_ocr_caches_and_filters_script(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.touch()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    ocr = FakeOcr()
    translator = FakeTranslator()
    pipeline = VisualTextPipeline(
        FakeSampler(image),
        FakeDetector(),
        ocr,
        translator,
        VisualEventTracker(frame_interval_seconds=0.5),
    )

    events = pipeline.process(media)

    assert len(events) == 1
    assert events[0].translated_text == "Japan"
    assert ocr.calls == 1
    assert translator.calls == 1
    pipeline.close()
    assert translator.calls == -1

    english_pipeline = VisualTextPipeline(
        FakeSampler(image),
        FakeDetector(),
        FakeOcr("English"),
        FakeTranslator(),
        VisualEventTracker(),
    )
    assert english_pipeline.process(media) == ()
    with pytest.raises(ValueError):
        VisualTextPipeline(
            FakeSampler(image),
            FakeDetector(),
            FakeOcr(),
            FakeTranslator(),
            VisualEventTracker(),
            crop_padding_ratio=0.51,
        )
    with pytest.raises(ValueError):
        VisualTextPipeline(
            FakeSampler(image),
            FakeDetector(),
            FakeOcr(),
            FakeTranslator(),
            VisualEventTracker(),
            minimum_box_area_ratio=1.01,
        )


def test_visual_pipeline_continues_when_batch_detection_crashes(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.touch()

    class BoomDetector:
        def detect_batch(self, _images: Any) -> tuple[tuple[BoundingBox, ...], ...]:
            raise NotImplementedError("onednn")

    events = VisualTextPipeline(
        FakeSampler(np.zeros((12, 12, 3), dtype=np.uint8)),
        BoomDetector(),
        FakeOcr(),
        FakeTranslator(),
        VisualEventTracker(frame_interval_seconds=0.5),
        region_proposer=FakeRegionProposer(),
        minimum_box_area_ratio=0.0,
        minimum_vertical_center_ratio=0.0,
        minimum_japanese_characters=1,
    ).process(media)
    assert events == ()


def test_visual_pipeline_pads_ocr_crops_within_frame_bounds(tmp_path: Path) -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    ocr = ShapeRecordingOcr()
    pipeline = VisualTextPipeline(
        FakeSampler(image),
        FakeDetector(),
        ocr,
        FakeTranslator(),
        VisualEventTracker(frame_interval_seconds=0.5),
        crop_padding_ratio=0.25,
        minimum_vertical_center_ratio=0.0,
    )

    pipeline.process(tmp_path / "video.mp4")

    assert ocr.shapes == [(10, 10, 3)]


def test_visual_pipeline_filters_small_and_upper_frame_regions(tmp_path: Path) -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    ocr = FakeOcr()
    pipeline = VisualTextPipeline(
        FakeSampler(image),
        FakeDetector(
            (
                BoundingBox(0, 0, 50, 20),
                BoundingBox(0, 70, 5, 5),
            )
        ),
        ocr,
        FakeTranslator(),
        VisualEventTracker(),
    )

    assert pipeline.process(tmp_path / "video.mp4") == ()
    assert ocr.calls == 0


def test_visual_pipeline_japanese_character_threshold_is_configurable(
    tmp_path: Path,
) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    media = tmp_path / "video.mp4"
    default_pipeline = VisualTextPipeline(
        FakeSampler(image),
        FakeDetector(),
        FakeOcr("日本語字"),
        FakeTranslator(),
        VisualEventTracker(frame_interval_seconds=0.5),
    )
    configured_pipeline = VisualTextPipeline(
        FakeSampler(image),
        FakeDetector(),
        FakeOcr("日本語字"),
        FakeTranslator(),
        VisualEventTracker(frame_interval_seconds=0.5),
        minimum_japanese_characters=4,
    )

    assert default_pipeline.process(media) == ()
    assert len(configured_pipeline.process(media)) == 1


def test_visual_pipeline_does_not_cache_perceptual_hash_collisions(
    tmp_path: Path,
) -> None:
    ocr = FakeOcr()
    pipeline = VisualTextPipeline(
        SequenceSampler(
            (
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.ones((8, 8, 3), dtype=np.uint8),
            )
        ),
        FakeDetector(),
        ocr,
        FakeTranslator(),
        VisualEventTracker(frame_interval_seconds=0.5),
    )

    pipeline.process(tmp_path / "video.mp4")

    assert ocr.calls == 2


def test_visual_pipeline_detects_only_proposed_regions_and_restores_coordinates(
    tmp_path: Path,
) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    detector = RecordingDetector()
    proposer = FakeRegionProposer()
    pipeline = VisualTextPipeline(
        FakeSampler(image),
        detector,
        FakeOcr(),
        FakeTranslator(),
        VisualEventTracker(frame_interval_seconds=0.5),
        region_proposer=proposer,
    )

    events = pipeline.process(tmp_path / "video.mp4")

    assert detector.shapes == [(416, 416, 3), (416, 416, 3)]
    assert events[0].box == BoundingBox(2, 2, 4, 4)
    assert proposer.resets == 1


def test_visual_pipeline_preserves_static_card_appearance_interval(
    tmp_path: Path,
) -> None:
    background = np.zeros((100, 200, 3), dtype=np.uint8)
    card = background.copy()
    card[40:80, 40:120] = 255

    def components(
        mask: np.ndarray[Any, Any],
    ) -> tuple[tuple[int, int, int, int, int], ...]:
        return ((8, 9, 8, 5, 40),) if mask.any() else ()

    pipeline = VisualTextPipeline(
        SequenceSampler((background, card, card, card, card, card, card)),
        BrightDetector(),
        FakeOcr(),
        FakeTranslator(),
        VisualEventTracker(frame_interval_seconds=0.5),
        region_proposer=TemporalDifferenceProposer(
            analysis_width=32,
            hold_frames=6,
            full_frame_hold_frames=1,
            padding_ratio=0,
            component_extractor=components,
        ),
    )

    events = pipeline.process(tmp_path / "video.mp4")

    assert len(events) == 1
    assert (events[0].start, events[0].end) == (0.5, 3.5)


def test_visual_pipeline_refreshes_opening_card_after_full_frame_hold(
    tmp_path: Path,
) -> None:
    card = np.full((100, 200, 3), 255, dtype=np.uint8)
    pipeline = VisualTextPipeline(
        SequenceSampler((card,) * 10),
        BrightDetector(),
        FakeOcr(),
        FakeTranslator(),
        VisualEventTracker(frame_interval_seconds=0.5),
        region_proposer=TemporalDifferenceProposer(
            component_extractor=lambda _mask: (),
        ),
    )

    events = pipeline.process(tmp_path / "video.mp4")

    assert len(events) == 1
    assert (events[0].start, events[0].end) == (0.0, 5.0)


def test_visual_pipeline_rechecks_perceptual_hash_collision(
    tmp_path: Path,
) -> None:
    black = np.zeros((100, 200, 3), dtype=np.uint8)
    white = np.full((100, 200, 3), 255, dtype=np.uint8)
    gray = np.full((100, 200, 3), 128, dtype=np.uint8)
    detector = BrightDetector()
    pipeline = VisualTextPipeline(
        SequenceSampler((black, white, white, white, gray, gray)),
        detector,
        FakeOcr(),
        FakeTranslator(),
        VisualEventTracker(frame_interval_seconds=0.5),
        region_proposer=FakeRegionProposer(),
        minimum_box_area_ratio=0,
        minimum_vertical_center_ratio=0,
    )

    pipeline.process(tmp_path / "video.mp4")

    assert perceptual_hash(white) == perceptual_hash(gray)
    assert detector.calls >= 5
