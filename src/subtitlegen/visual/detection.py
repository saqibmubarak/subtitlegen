from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from subtitlegen.errors import BackendUnavailableError
from subtitlegen.visual.models import BoundingBox

logger = logging.getLogger(__name__)

_ONEDNN_ENV = (
    "FLAGS_use_mkldnn",
    "FLAGS_enable_mkldnn",
    "FLAGS_enable_pir_api",
    "FLAGS_enable_pir_in_executor",
)


def disable_paddle_onednn() -> None:
    """PaddleX defaults CPU detection to OneDNN, which crashes Paddle 3.3 PIR."""
    for flag in _ONEDNN_ENV:
        os.environ[flag] = "0"
    try:
        import paddle
    except ImportError:
        return
    setter = getattr(paddle, "set_flags", None)
    if setter is not None:
        try:
            setter({"FLAGS_use_mkldnn": False})
        except (RuntimeError, TypeError, ValueError):
            return


def text_detection_options(*, device_type: str = "cpu") -> dict[str, Any]:
    if device_type not in {"cpu", "gpu"}:
        raise ValueError("Paddle device type must be cpu or gpu")
    return {
        "model_name": "PP-OCRv5_mobile_det",
        "thresh": 0.3,
        "box_thresh": 0.5,
        "unclip_ratio": 1.35,
        "enable_mkldnn": False,
        "engine": "paddle_static",
        "engine_config": {
            "device_type": device_type,
            "run_mode": "paddle",
            "enable_new_ir": False,
        },
    }


def text_recognition_options(*, device_type: str = "cpu") -> dict[str, Any]:
    if device_type not in {"cpu", "gpu"}:
        raise ValueError("Paddle device type must be cpu or gpu")
    return {
        "model_name": "PP-OCRv5_mobile_rec",
        "enable_mkldnn": False,
        "engine": "paddle_static",
        "engine_config": {
            "device_type": device_type,
            "run_mode": "paddle",
            "enable_new_ir": False,
        },
    }


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
    def __init__(
        self,
        *,
        engine_factory: Callable[[], Any] | None = None,
        runtime: Any | None = None,
    ) -> None:
        self._engine_factory = engine_factory
        self._runtime = runtime
        self._engine: Any | None = None

    def detect(self, image: Any) -> tuple[BoundingBox, ...]:
        if self._runtime is not None:
            return tuple(self._runtime.detect(image))
        engine = self._load_engine()
        if hasattr(engine, "predict"):
            try:
                result = engine.predict(np.asarray(image))
            except NotImplementedError:
                logger.warning("Paddle text detection is unavailable on this runtime")
                return ()
            return self._boxes_from_payload(result[0] if result else {})
        result = engine.ocr(np.asarray(image), det=True, rec=False, cls=False)
        regions = result[0] if result and len(result) == 1 else result
        confidences = [1.0] * len(regions or ())
        return self._boxes(regions or (), confidences)

    def detect_batch(self, images: Sequence[Any]) -> tuple[tuple[BoundingBox, ...], ...]:
        if self._runtime is not None:
            return tuple(self._runtime.detect_batch(images))
        engine = self._load_engine()
        if not hasattr(engine, "predict"):
            return tuple(self.detect(image) for image in images)
        try:
            results = engine.predict(
                [np.asarray(image) for image in images],
                batch_size=min(16, len(images)),
            )
        except NotImplementedError:
            logger.warning("Paddle batch detection fell back after an OneDNN error")
            self.close()
            disable_paddle_onednn()
            try:
                return tuple(self.detect(image) for image in images)
            except NotImplementedError:
                return tuple(() for _ in images)
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
        close = getattr(self._runtime, "close", None)
        if close is not None:
            close()
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
                disable_paddle_onednn()
                options = text_detection_options()
                try:
                    return TextDetection(**options)
                except TypeError:
                    options.pop("engine_config", None)
                    options.pop("engine", None)
                    try:
                        return TextDetection(**options)
                    except TypeError:
                        options.pop("enable_mkldnn", None)
                        return TextDetection(**options)

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

    def detect_batch(
        self,
        images: Sequence[Any],
    ) -> tuple[tuple[BoundingBox, ...], ...]:
        primary_batch = getattr(self._primary, "detect_batch", None)
        if primary_batch is None:
            return tuple(tuple(self.detect(image)) for image in images)
        try:
            primary_results = tuple(tuple(boxes) for boxes in primary_batch(images))
        except (BackendUnavailableError, RuntimeError):
            primary_results = tuple(() for _ in images)
        if len(primary_results) != len(images):
            raise RuntimeError("primary text detector returned an incomplete batch")
        missing = [index for index, boxes in enumerate(primary_results) if not boxes]
        if not missing:
            return primary_results
        fallback_images = [images[index] for index in missing]
        fallback_batch = getattr(self._fallback, "detect_batch", None)
        fallback_results = (
            fallback_batch(fallback_images)
            if fallback_batch is not None
            else tuple(self._fallback.detect(image) for image in fallback_images)
        )
        results = list(primary_results)
        for index, boxes in zip(missing, fallback_results, strict=True):
            results[index] = tuple(boxes)
        return tuple(results)

    def close(self) -> None:
        for detector in (self._primary, self._fallback):
            close = getattr(detector, "close", None)
            if close is not None:
                close()
