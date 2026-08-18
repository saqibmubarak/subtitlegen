from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from subtitlegen.errors import BackendUnavailableError
from subtitlegen.visual.models import BoundingBox


class TextDetector(Protocol):
    def detect(self, image: Any) -> Sequence[BoundingBox]:
        """Return candidate text regions without performing OCR."""


class OpenCvDbNetDetector:
    """Adapter for anime/comic-trained DBNet ONNX text detectors."""

    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float = 0.5,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not 0 < confidence_threshold <= 1:
            raise ValueError("detector confidence threshold must be in (0, 1]")
        if model_factory is None and not model_path.is_file():
            raise FileNotFoundError(model_path)
        self._model_path = model_path
        self._threshold = confidence_threshold
        self._model_factory = model_factory
        self._model: Any | None = None

    def detect(self, image: Any) -> tuple[BoundingBox, ...]:
        detected = self._load_model().detect(np.asarray(image))
        if len(detected) == 2:
            regions, confidences = detected
        else:
            _, confidences, regions = detected
        boxes = [
            self._to_box(region, float(confidence))
            for region, confidence in zip(regions, confidences, strict=True)
            if float(confidence) >= self._threshold
        ]
        return tuple(box for box in boxes if box is not None)

    def close(self) -> None:
        self._model = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        factory = self._model_factory
        if factory is None:
            try:
                import cv2
            except ImportError as error:
                raise BackendUnavailableError(
                    "DBNet detection requires subtitlegen[ocr]"
                ) from error
            factory = cv2.dnn_TextDetectionModel_DB  # type: ignore[attr-defined]
        model = factory(str(self._model_path))
        if hasattr(model, "setBinaryThreshold"):
            model.setBinaryThreshold(self._threshold)
            model.setPolygonThreshold(self._threshold)
            model.setUnclipRatio(2.0)
            model.setInputParams(
                scale=1 / 255,
                size=(736, 736),
                mean=(122.67891434, 116.66876762, 104.00698793),
                swapRB=True,
            )
        self._model = model
        return model

    @staticmethod
    def _to_box(region: Any, confidence: float) -> BoundingBox | None:
        array = np.asarray(region, dtype=np.int64)
        if array.size == 4 and array.ndim == 1:
            x, y, width, height = (int(value) for value in array)
        else:
            points = array.reshape(-1, 2)
            x = max(0, int(points[:, 0].min()))
            y = max(0, int(points[:, 1].min()))
            width = int(points[:, 0].max()) - x + 1
            height = int(points[:, 1].max()) - y + 1
        if width <= 0 or height <= 0:
            return None
        return BoundingBox(max(0, x), max(0, y), width, height, confidence)


class PaddleOcrDetector:
    def __init__(self, *, engine_factory: Callable[[], Any] | None = None) -> None:
        self._engine_factory = engine_factory
        self._engine: Any | None = None

    def detect(self, image: Any) -> tuple[BoundingBox, ...]:
        engine = self._load_engine()
        if hasattr(engine, "predict"):
            result = engine.predict(np.asarray(image))
            return self._boxes_from_payload(result[0] if result else {})
        else:
            result = engine.ocr(np.asarray(image), det=True, rec=False, cls=False)
            regions = result[0] if result and len(result) == 1 else result
            confidences = [1.0] * len(regions or ())
            return self._boxes(regions or (), confidences)

    def detect_batch(self, images: Sequence[Any]) -> tuple[tuple[BoundingBox, ...], ...]:
        engine = self._load_engine()
        if not hasattr(engine, "predict"):
            return tuple(self.detect(image) for image in images)
        results = engine.predict(
            [np.asarray(image) for image in images],
            batch_size=min(16, len(images)),
        )
        return tuple(self._boxes_from_payload(payload) for payload in results)

    @staticmethod
    def _boxes_from_payload(payload: Any) -> tuple[BoundingBox, ...]:
        regions = payload.get("dt_polys", ())
        confidences = payload.get("dt_scores", [1.0] * len(regions))
        return PaddleOcrDetector._boxes(regions, confidences)

    @staticmethod
    def _boxes(regions: Any, confidences: Any) -> tuple[BoundingBox, ...]:
        boxes = [
            OpenCvDbNetDetector._to_box(region, float(confidence))
            for region, confidence in zip(regions, confidences, strict=True)
        ]
        return tuple(box for box in boxes if box is not None)

    def close(self) -> None:
        self._engine = None

    def _load_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        factory = self._engine_factory
        if factory is None:
            try:
                from paddleocr import TextDetection
            except ImportError as error:
                raise BackendUnavailableError(
                    "PaddleOCR fallback requires the OCR container dependencies"
                ) from error

            def factory() -> Any:
                return TextDetection(
                    model_name="PP-OCRv5_mobile_det",
                    thresh=0.3,
                    box_thresh=0.5,
                    unclip_ratio=2.0,
                )

        self._engine = factory()
        return self._engine


class FallbackTextDetector:
    def __init__(self, primary: TextDetector, fallback: TextDetector) -> None:
        self._primary = primary
        self._fallback = fallback

    def detect(self, image: Any) -> Sequence[BoundingBox]:
        try:
            detected = self._primary.detect(image)
        except (BackendUnavailableError, RuntimeError):
            detected = ()
        return detected or self._fallback.detect(image)

    def close(self) -> None:
        for detector in (self._primary, self._fallback):
            close = getattr(detector, "close", None)
            if close is not None:
                close()
