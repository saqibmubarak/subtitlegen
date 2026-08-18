from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from subtitlegen.asr.capabilities import BackendCapabilities
from subtitlegen.asr.context import AsrContext
from subtitlegen.domain.models import Transcription, Word
from subtitlegen.errors import BackendOutOfMemoryError, BackendUnavailableError
from subtitlegen.settings import AsrSettings

ModelFactory = Callable[[str], Any]


class ParakeetBackend:
    """English NVIDIA Parakeet TDT adapter with native word timestamps."""

    DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"

    def __init__(
        self,
        settings: AsrSettings,
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self._settings = settings
        self._model_factory = model_factory
        self._model: Any | None = None

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(True, False, False, True)

    def transcribe(
        self,
        media_path: Path,
        *,
        language: str | None = None,
        context: AsrContext | None = None,
    ) -> Transcription:
        if not media_path.is_file():
            raise FileNotFoundError(media_path)
        requested_language = language or self._settings.language or "en"
        if requested_language.split("-")[0].casefold() != "en":
            raise ValueError("Parakeet TDT 0.6B v3 supports English audio only")
        if context is not None:
            raise BackendUnavailableError(
                "Parakeet context boosting is unavailable in this adapter; "
                "use faster-whisper or WhisperX for series profiles"
            )
        try:
            hypotheses = self._load_model().transcribe(
                [str(media_path)],
                batch_size=1,
                timestamps=True,
            )
        except RuntimeError as error:
            if "out of memory" in str(error).casefold():
                raise BackendOutOfMemoryError(
                    "Parakeet exhausted VRAM; close other GPU models or use the fast preset"
                ) from error
            raise
        hypothesis = hypotheses[0]
        timestamp = getattr(hypothesis, "timestamp", {}) or {}
        items = timestamp.get("word", ())
        words = sorted(
            (
                Word(
                    start=max(0.0, float(item["start"])),
                    end=max(0.0, float(item["start"]), float(item["end"])),
                    text=str(item.get("word") or item.get("char") or ""),
                )
                for item in items
                if item.get("start") is not None
                and item.get("end") is not None
                and str(item.get("word") or item.get("char") or "").strip()
            ),
            key=lambda word: (word.start, word.end),
        )
        return Transcription(
            words=tuple(words),
            language="en",
            duration=max((word.end for word in words), default=0.0),
        )

    def close(self) -> None:
        self._model = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        factory = self._model_factory
        if factory is None:
            try:
                from nemo.collections.asr.models import ASRModel
            except ImportError as error:
                raise BackendUnavailableError(
                    "Parakeet is unavailable; install subtitlegen[nemo]"
                ) from error

            def factory(name: str) -> Any:
                return ASRModel.from_pretrained(model_name=name)

        model_name = self._settings.model
        if model_name in {"large-v3", "large-v3-turbo", "turbo"}:
            model_name = self.DEFAULT_MODEL
        self._model = factory(model_name)
        return self._model
