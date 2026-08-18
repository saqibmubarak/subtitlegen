from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from subtitlegen.asr.context import AsrContext
from subtitlegen.domain.models import Transcription, Word
from subtitlegen.settings import AsrSettings

ModelFactory = Callable[..., Any]


class FasterWhisperBackend:
    """Normalized faster-whisper adapter with one model per backend instance."""

    def __init__(
        self,
        settings: AsrSettings,
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self._settings = settings
        self._model_factory = model_factory
        self._model: Any | None = None

    def transcribe(
        self,
        media_path: Path,
        *,
        language: str | None = None,
        context: AsrContext | None = None,
    ) -> Transcription:
        if not media_path.exists():
            raise FileNotFoundError(media_path)
        model = self._load_model()
        segments, info = model.transcribe(
            str(media_path),
            language=language if language is not None else self._settings.language,
            beam_size=self._settings.beam_size,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": self._settings.vad.min_silence_duration_ms,
                "speech_pad_ms": self._settings.vad.speech_pad_ms,
                "max_speech_duration_s": self._settings.vad.max_speech_duration_s,
            },
            word_timestamps=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
            initial_prompt=context.prompt if context is not None else None,
            hotwords=" ".join(context.hotwords) if context is not None else None,
        )

        words: list[Word] = []
        for segment in segments:
            for item in segment.words or ():
                if item.start is None or item.end is None or not item.word.strip():
                    continue
                words.append(
                    Word(
                        start=max(0.0, float(item.start)),
                        end=max(float(item.start), float(item.end)),
                        text=str(item.word),
                        probability=getattr(item, "probability", None),
                    )
                )

        words.sort(key=lambda word: (word.start, word.end))
        return Transcription(
            words=tuple(words),
            language=getattr(info, "language", None) or language or "unknown",
            duration=getattr(info, "duration", None),
        )

    def close(self) -> None:
        self._model = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        device = self._resolve_device(self._settings.device)
        compute_type = self._settings.compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        factory = self._model_factory
        if factory is None:
            from faster_whisper import WhisperModel

            factory = WhisperModel
        self._model = factory(
            self._settings.model,
            device=device,
            compute_type=compute_type,
        )
        return self._model

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested not in {"auto", "cuda"}:
            return requested
        try:
            import ctranslate2

            cuda_available = ctranslate2.get_cuda_device_count() > 0
        except (ImportError, RuntimeError):
            cuda_available = False
        if requested == "cuda" and not cuda_available:
            return "cpu"
        return "cuda" if cuda_available else "cpu"
