from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from subtitlegen.asr.capabilities import BackendCapabilities
from subtitlegen.asr.context import AsrContext
from subtitlegen.asr.whisper_context import fit_whisper_context, is_decoder_overflow
from subtitlegen.domain.models import Transcription, Word
from subtitlegen.errors import BackendOutOfMemoryError, BackendUnavailableError
from subtitlegen.settings import AsrSettings

logger = logging.getLogger(__name__)

# Leave room for sot/language/task tokens inside the 448-position decoder.
_MAX_NEW_TOKENS = 224


class WhisperXBackend:
    """CUDA Whisper transcription plus forced word alignment."""

    def __init__(
        self,
        settings: AsrSettings,
        *,
        api: Any | None = None,
        batch_size: int | None = None,
        fallback: Any | None = None,
    ) -> None:
        effective_batch_size = settings.whisperx_batch_size if batch_size is None else batch_size
        if effective_batch_size <= 0:
            raise ValueError("WhisperX batch size must be positive")
        self._settings = settings
        self._api = api
        self._batch_size = effective_batch_size
        self._fallback = fallback
        self._model: Any | None = None
        self._loaded_context: AsrContext | None = None
        self._aligners: dict[str, tuple[Any, Any]] = {}

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(True, True, True, True)

    def transcribe(
        self,
        media_path: Path,
        *,
        language: str | None = None,
        context: AsrContext | None = None,
    ) -> Transcription:
        if not media_path.is_file():
            raise FileNotFoundError(media_path)
        fitted = fit_whisper_context(context)
        try:
            return self._transcribe_aligned(media_path, language=language, context=fitted)
        except RuntimeError as error:
            if "out of memory" in str(error).casefold():
                raise BackendOutOfMemoryError(
                    "WhisperX exhausted VRAM; lower TRANSCRIPTION.whisperx_batch_size "
                    "or use --preset fast"
                ) from error
            if not is_decoder_overflow(error):
                raise
            if fitted is not None:
                logger.warning(
                    "WhisperX prompt exceeded the 448-token decoder window; "
                    "retrying without glossary context"
                )
                self._drop_model()
                try:
                    return self._transcribe_aligned(media_path, language=language, context=None)
                except RuntimeError as retry_error:
                    if "out of memory" in str(retry_error).casefold():
                        raise BackendOutOfMemoryError(
                            "WhisperX exhausted VRAM; lower TRANSCRIPTION.whisperx_batch_size "
                            "or use --preset fast"
                        ) from retry_error
                    if not is_decoder_overflow(retry_error):
                        raise
                    error = retry_error
            logger.warning(
                "WhisperX hit the 448-token decoder limit; falling back to faster-whisper"
            )
            self._drop_model()
            return self._fallback_backend().transcribe(
                media_path,
                language=language,
                context=fitted,
            )

    def close(self) -> None:
        self._drop_model()
        if self._fallback is not None:
            close = getattr(self._fallback, "close", None)
            if close is not None:
                close()

    def _transcribe_aligned(
        self,
        media_path: Path,
        *,
        language: str | None,
        context: AsrContext | None,
    ) -> Transcription:
        api = self._load_api()
        device = "cuda" if self._settings.device == "auto" else self._settings.device
        if device != "cuda":
            raise BackendUnavailableError("WhisperX alignment requires a CUDA device")
        model = self._load_model(api, context)
        audio = api.load_audio(str(media_path))
        result = model.transcribe(
            audio,
            batch_size=self._batch_size,
            language=language if language is not None else self._settings.language,
        )
        detected_language = str(result.get("language") or language or "unknown")
        aligner, metadata = self._aligners.get(detected_language, (None, None))
        if aligner is None:
            aligner, metadata = api.load_align_model(
                language_code=detected_language,
                device=device,
            )
            self._aligners[detected_language] = (aligner, metadata)
        aligned = api.align(
            result["segments"],
            aligner,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        return self._transcription(aligned, detected_language)

    def _load_api(self) -> Any:
        if self._api is not None:
            return self._api
        try:
            import whisperx
        except ImportError as error:
            raise BackendUnavailableError(
                "WhisperX is unavailable; install subtitlegen[cuda]"
            ) from error
        self._api = whisperx
        return whisperx

    def _load_model(self, api: Any, context: AsrContext | None) -> Any:
        if self._model is not None and self._loaded_context != context:
            self._drop_model()
        if self._model is None:
            compute_type = (
                "float16" if self._settings.compute_type == "auto" else self._settings.compute_type
            )
            self._model = api.load_model(
                self._settings.model,
                "cuda",
                compute_type=compute_type,
                asr_options=self._asr_options(context),
            )
            self._loaded_context = context
        return self._model

    def _drop_model(self) -> None:
        self._model = None
        self._loaded_context = None
        self._aligners.clear()

    def _fallback_backend(self) -> Any:
        if self._fallback is None:
            from subtitlegen.asr.faster_whisper import FasterWhisperBackend

            self._fallback = FasterWhisperBackend(self._settings)
        return self._fallback

    @staticmethod
    def _asr_options(context: AsrContext | None) -> dict[str, Any]:
        return {
            "initial_prompt": context.prompt if context is not None else None,
            "hotwords": (
                " ".join(context.hotwords)
                if context is not None and context.hotwords
                else None
            ),
            "max_new_tokens": _MAX_NEW_TOKENS,
        }

    @staticmethod
    def _transcription(aligned: dict[str, Any], language: str) -> Transcription:
        words: list[Word] = []
        items = aligned.get("word_segments")
        if items is None:
            items = [
                word
                for segment in aligned.get("segments", [])
                for word in segment.get("words", [])
            ]
        for item in items:
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
                    probability=item.get("score"),
                )
            )
        words.sort(key=lambda word: (word.start, word.end))
        return Transcription(
            tuple(words),
            language,
            max((word.end for word in words), default=0.0),
        )
