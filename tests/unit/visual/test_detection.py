from pathlib import Path
from typing import Any

import numpy as np
import pytest

from subtitlegen.visual.detection import (
    FallbackTextDetector,
    OpenCvDbNetDetector,
    PaddleOcrDetector,
)
from subtitlegen.visual.models import BoundingBox


class FakeDbModel:
    def __init__(self) -> None:
        self.configured = 0

    def setBinaryThreshold(self, _value: float) -> None:
        self.configured += 1

    def setPolygonThreshold(self, _value: float) -> None:
        self.configured += 1

    def setUnclipRatio(self, _value: float) -> None:
        self.configured += 1

    def setInputParams(self, **_kwargs: Any) -> None:
        self.configured += 1

    def detect(self, _image: Any) -> tuple[list[Any], list[float]]:
        return (
            [
                np.array([[1, 2], [5, 2], [5, 6], [1, 6]]),
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            ],
            [0.9, 0.1],
        )


class FakePaddle:
    def ocr(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return [[[[2, 3], [8, 3], [8, 9], [2, 9]]]]


class FakePaddlePredict:
    def predict(self, images: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        items = images if isinstance(images, list) else [images]
        return [
            {
                "dt_polys": [np.array([[2, 3], [8, 3], [8, 9], [2, 9]])],
                "dt_scores": [0.8],
            }
            for _ in items
        ]


def test_dbnet_detector_configures_filters_normalizes_and_releases(tmp_path: Path) -> None:
    model = FakeDbModel()
    detector = OpenCvDbNetDetector(
        tmp_path / "model.onnx",
        confidence_threshold=0.5,
        model_factory=lambda _path: model,
    )
    assert detector.detect(np.zeros((10, 10, 3))) == (BoundingBox(1, 2, 5, 5, 0.9),)
    assert model.configured == 4
    detector.detect(np.zeros((10, 10, 3)))
    assert model.configured == 4
    detector.close()
    with pytest.raises(ValueError):
        OpenCvDbNetDetector(tmp_path / "model", confidence_threshold=0)
    with pytest.raises(FileNotFoundError):
        OpenCvDbNetDetector(tmp_path / "missing")


def test_paddle_detector_and_fallback_contract() -> None:
    paddle = PaddleOcrDetector(engine_factory=FakePaddle)
    result = paddle.detect(np.zeros((10, 10, 3)))
    assert result == (BoundingBox(2, 3, 7, 7),)
    paddle.close()
    batch_detector = PaddleOcrDetector(engine_factory=FakePaddlePredict)
    assert len(batch_detector.detect_batch([np.zeros((10, 10)), np.zeros((10, 10))])) == 2

    class EmptyDetector:
        def detect(self, _image: Any) -> tuple[BoundingBox, ...]:
            return ()

    fallback = FallbackTextDetector(EmptyDetector(), paddle)
    assert fallback.detect(np.zeros((10, 10, 3))) == result
    fallback.close()

    class PartialBatchDetector:
        def detect(self, _image: Any) -> tuple[BoundingBox, ...]:
            return ()

        def detect_batch(
            self,
            _images: Any,
        ) -> tuple[tuple[BoundingBox, ...], ...]:
            return ((BoundingBox(1, 1, 2, 2),), ())

    batched_fallback = FallbackTextDetector(PartialBatchDetector(), batch_detector)
    batched = batched_fallback.detect_batch(
        [np.zeros((10, 10)), np.zeros((10, 10))]
    )
    assert batched[0] == (BoundingBox(1, 1, 2, 2),)
    assert batched[1] == (BoundingBox(2, 3, 7, 7, 0.8),)


@pytest.mark.parametrize(
    "detector",
    [
        OpenCvDbNetDetector(
            Path("model.onnx"),
            model_factory=lambda _path: FakeDbModel(),
        ),
        PaddleOcrDetector(engine_factory=FakePaddle),
    ],
    ids=["dbnet", "paddle"],
)
def test_detector_contract_returns_valid_boxes(detector: Any) -> None:
    boxes = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert boxes
    assert all(box.width > 0 and box.height > 0 for box in boxes)
