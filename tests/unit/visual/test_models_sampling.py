from pathlib import Path

import numpy as np
import pytest

from subtitlegen.visual.models import (
    BoundingBox,
    OcrResult,
    SampledFrame,
    StyledCue,
    VisualEvent,
    VisualObservation,
)
from subtitlegen.visual.sampler import FrameSampler
from subtitlegen.visual.tracker import perceptual_hash


def test_visual_domain_models_validate_and_box_iou() -> None:
    box = BoundingBox(0, 0, 10, 10, 0.8)
    assert box.area == 100
    assert box.intersection_over_union(BoundingBox(5, 0, 10, 10)) == pytest.approx(1 / 3)
    SampledFrame(0, object())
    OcrResult("日本", 0.9)
    observation = VisualObservation(0, box, "日本", "Japan", 1)
    assert observation.translated_text == "Japan"
    VisualEvent(0, 1, "日本", "Japan", box)
    StyledCue(0, 1, "Japan", "OnScreen")

    invalid = [
        lambda: BoundingBox(-1, 0, 1, 1),
        lambda: BoundingBox(0, 0, 0, 1),
        lambda: BoundingBox(0, 0, 1, 1, 2),
        lambda: SampledFrame(-1, object()),
        lambda: OcrResult("text", 2),
        lambda: VisualObservation(0, box, "", "Japan", 1),
        lambda: VisualEvent(1, 1, "日本", "Japan", box),
        lambda: StyledCue(0, 1, "", "OnScreen"),
    ]
    for create in invalid:
        with pytest.raises(ValueError):
            create()


def test_frame_sampler_combines_regular_and_scene_change_frames(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.touch()
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    white = np.full((32, 32, 3), 255, dtype=np.uint8)
    frames = [(0.0, black), (0.2, black), (0.7, white), (1.1, white)]
    sampler = FrameSampler(
        frames_per_second=1,
        scene_threshold=0.2,
        frame_reader=lambda _path: frames,
    )

    sampled = tuple(sampler.sample(media))

    assert [frame.timestamp for frame in sampled] == [0.0, 0.7, 1.1]
    assert sampled[1].scene_change
    with pytest.raises(FileNotFoundError):
        tuple(sampler.sample(tmp_path / "missing.mp4"))
    with pytest.raises(ValueError):
        FrameSampler(frames_per_second=3)
    with pytest.raises(ValueError):
        FrameSampler(scene_threshold=0)


def test_perceptual_hash_is_deterministic_and_rejects_empty_images() -> None:
    image = np.arange(81, dtype=np.uint8).reshape(9, 9)
    assert perceptual_hash(image) == perceptual_hash(image.copy())
    with pytest.raises(ValueError):
        perceptual_hash(np.array([]))
