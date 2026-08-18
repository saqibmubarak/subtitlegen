from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from subtitlegen.errors import BackendUnavailableError
from subtitlegen.visual.models import OcrResult

JAPANESE_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")


class OcrEngine(Protocol):
    def recognize(self, image: Any) -> OcrResult:
        """Recognize one previously detected text crop."""


def contains_japanese(text: str) -> bool:
    return JAPANESE_PATTERN.search(text) is not None


def japanese_character_count(text: str) -> int:
    return len(JAPANESE_PATTERN.findall(text))


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
