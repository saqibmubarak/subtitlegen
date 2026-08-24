from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from subtitlegen.errors import BackendUnavailableError
from subtitlegen.visual.models import OcrResult

logger = logging.getLogger(__name__)


def warmup_torch() -> None:
    """Allocate one PyTorch tensor before Paddle imports.

    Paddle's allocator can break later ``torch.randn`` calls on this runtime.
    """
    import torch

    torch.zeros(1)

JAPANESE_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
HIRAGANA_PATTERN = re.compile(r"[\u3040-\u309f]")
KANJI_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
KATAKANA_PATTERN = re.compile(r"[\u30a0-\u30ff]")


class OcrEngine(Protocol):
    def recognize(self, image: Any) -> OcrResult:
        """Recognize one previously detected text crop."""


def contains_japanese(text: str) -> bool:
    return JAPANESE_PATTERN.search(text) is not None


def rotate_vertical_crop(image: Any) -> Any:
    """Turn a tate-gaki crop into a left-to-right strip for horizontal OCR."""
    array = np.asarray(image)
    if array.ndim < 2 or array.shape[0] <= array.shape[1]:
        return array
    return np.rot90(array, k=-1)


def japanese_character_count(text: str) -> int:
    return len(JAPANESE_PATTERN.findall(text))


def hiragana_character_count(text: str) -> int:
    return len(HIRAGANA_PATTERN.findall(text))


def kanji_character_count(text: str) -> int:
    return len(KANJI_PATTERN.findall(text))


def katakana_character_count(text: str) -> int:
    return len(KATAKANA_PATTERN.findall(text))


def has_title_script(
    text: str,
    *,
    minimum_kanji: int = 2,
    minimum_katakana: int = 3,
) -> bool:
    """Keep location/name cards; drop Manga OCR hiragana filler.

    Kanji are the Chinese-origin characters used in names and place titles
    (``一人``, ``錦``). Katakana is the angular script used for names like
    ``ドレスローザ``. Conversational hiragana-only dumps such as ``そういえば``
    fail both thresholds. A single kanji in filler (``人のところで``) also fails.
    """
    return (
        kanji_character_count(text) >= minimum_kanji
        or katakana_character_count(text) >= minimum_katakana
    )


class PaddleTextRecognizer:
    """Fast mobile recognizer used to decide whether a frame has Japanese text."""

    def __init__(
        self,
        *,
        engine_factory: Callable[[], Any] | None = None,
        runtime: Any | None = None,
    ) -> None:
        self._engine_factory = engine_factory
        self._runtime = runtime
        self._engine: Any | None = None

    def recognize(self, image: Any) -> OcrResult:
        if self._runtime is not None:
            return OcrResult(str(self._runtime.recognize(image)))
        engine = self._load_engine()
        payload = self._predict(engine, np.asarray(image))
        return OcrResult(self._text_from_payload(payload))

    def close(self) -> None:
        close = getattr(self._runtime, "close", None)
        if close is not None:
            close()
        engine_close = getattr(self._engine, "close", None)
        if engine_close is not None:
            engine_close()
        self._engine = None

    def _predict(self, engine: Any, image: Any) -> Any:
        if hasattr(engine, "predict"):
            try:
                result = engine.predict(image)
            except NotImplementedError:
                logger.warning("Paddle text recognition is unavailable on this runtime")
                return {}
            if isinstance(result, list):
                return result[0] if result else {}
            return result or {}
        result = engine.ocr(image, det=False, rec=True, cls=False)
        return result[0] if result else {}

    @staticmethod
    def _text_from_payload(payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, dict):
            for key in ("rec_text", "text"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""
        if isinstance(payload, (list, tuple)) and payload:
            first = payload[0]
            if isinstance(first, str):
                return first.strip()
            if isinstance(first, dict):
                return PaddleTextRecognizer._text_from_payload(first)
            if isinstance(first, (list, tuple)) and first and isinstance(first[0], str):
                return first[0].strip()
            return ""
        return ""

    def _load_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        factory = self._engine_factory
        if factory is None:
            try:
                from paddleocr import TextRecognition
            except ImportError as error:
                raise BackendUnavailableError(
                    "PaddleOCR recognition requires the OCR container dependencies"
                ) from error

            def factory() -> Any:
                from subtitlegen.visual.detection import (
                    disable_paddle_onednn,
                    text_recognition_options,
                )

                disable_paddle_onednn()
                options = text_recognition_options()
                try:
                    return TextRecognition(**options)
                except TypeError:
                    options.pop("engine_config", None)
                    options.pop("engine", None)
                    try:
                        return TextRecognition(**options)
                    except TypeError:
                        options.pop("enable_mkldnn", None)
                        return TextRecognition(**options)

        self._engine = factory()
        return self._engine


class MangaOcrEngine:
    def __init__(
        self,
        *,
        model_factory: Callable[[], Any] | None = None,
        image_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._image_factory = image_factory
        self._model: Any | None = None

    def warmup(self) -> None:
        """Load PyTorch weights before Paddle runs; later torch inits can fail."""
        self._load_model()

    def recognize(self, image: Any) -> OcrResult:
        model = self._load_model()
        prepared = self._prepare_image(image)
        return OcrResult(str(model(prepared)).strip())

    def close(self) -> None:
        self._model = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        factory = self._model_factory
        if factory is None:
            try:
                from manga_ocr import MangaOcr
            except ImportError as error:
                raise BackendUnavailableError(
                    "manga-ocr is unavailable; install subtitlegen[ocr]"
                ) from error
            factory = MangaOcr
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*use_fast.*fast version.*",
            )
            self._model = factory()
        return self._model

    def _prepare_image(self, image: Any) -> Any:
        if self._image_factory is not None:
            return self._image_factory(image)
        try:
            from PIL import Image
        except ImportError as error:
            raise BackendUnavailableError("manga-ocr requires subtitlegen[ocr]") from error
        return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
