from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from subtitlegen.asr.capabilities import BackendCapabilities
from subtitlegen.asr.context import AsrContext
from subtitlegen.domain.models import Transcription, Word
from subtitlegen.errors import BackendOutOfMemoryError, BackendUnavailableError
from subtitlegen.media import load_audio_mono
from subtitlegen.settings import AsrSettings

logger = logging.getLogger(__name__)

MlxTranscribe = Callable[..., dict[str, Any]]
AudioLoader = Callable[[Path], Any]


class MlxWhisperBackend:
    """Apple Silicon Whisper adapter using mlx-whisper."""

    def __init__(
        self,
        settings: AsrSettings,
        *,
        transcribe_fn: MlxTranscribe | None = None,
        audio_loader: AudioLoader = load_audio_mono,
    ) -> None:
        self._settings = settings
        self._transcribe_fn = transcribe_fn
        self._audio_loader = audio_loader

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            word_timestamps=True,
            context_prompt=True,
            hotwords=False,
            requires_cuda=False,
        )

    def transcribe(
        self,
        media_path: Path,
        *,
        language: str | None = None,
        context: AsrContext | None = None,
    ) -> Transcription:
        if not media_path.exists():
            raise FileNotFoundError(media_path)
        transcribe = self._transcribe_fn
        if transcribe is None:
            try:
                import mlx_whisper
            except ImportError as error:
                raise BackendUnavailableError(
                    "MLX backend is unavailable; install subtitlegen[mac]"
                ) from error
            transcribe = mlx_whisper.transcribe

        audio = self._audio_loader(media_path)
        sample_rate = 16_000
        try:
            result = transcribe(
                audio,
                path_or_hf_repo=self._model_repository(self._settings.model),
                language=language if language is not None else self._settings.language,
                word_timestamps=True,
                condition_on_previous_text=False,
                hallucination_silence_threshold=2.0,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
                initial_prompt=context.prompt if context is not None else None,
            )
        except RuntimeError as error:
            if "out of memory" in str(error).casefold():
                raise BackendOutOfMemoryError(
                    "MLX Whisper exhausted memory; use the fast preset or a smaller model"
                ) from error
            raise
        words: list[Word] = []
        for segment in result.get("segments", []):
            for item in segment.get("words", []):
                start = item.get("start")
                end = item.get("end")
                text = str(item.get("word", ""))
                if start is None or end is None or not text.strip():
                    continue
                normalized_start = max(0.0, float(start))
                words.append(
                    Word(
                        start=normalized_start,
                        end=max(normalized_start, float(end)),
                        text=text,
                        probability=item.get("probability"),
                    )
                )
        words.sort(key=lambda word: (word.start, word.end))
        duration = len(audio) / sample_rate
        logger.info(
            "asr-mlx model=%s duration=%.1fs segments=%d words=%d language=%s",
            self._settings.model,
            duration,
            len(result.get("segments", [])),
            len(words),
            result.get("language") or language or "unknown",
        )
        return Transcription(
            words=tuple(words),
            language=str(result.get("language") or language or "unknown"),
            duration=duration,
        )

    def close(self) -> None:
        """MLX Whisper owns no persistent model object in this adapter."""

    @staticmethod
    def _model_repository(model: str) -> str:
        if "/" in model:
            return model
        aliases = {
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "turbo": "mlx-community/whisper-large-v3-turbo",
        }
        return aliases.get(model, f"mlx-community/whisper-{model}-mlx")
