from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from subtitlegen.asr.capabilities import BackendCapabilities
from subtitlegen.asr.context import AsrContext
from subtitlegen.domain.models import Transcription, Word
from subtitlegen.errors import BackendOutOfMemoryError, BackendUnavailableError
from subtitlegen.media import load_audio_mono
from subtitlegen.settings import AsrSettings

logger = logging.getLogger(__name__)

ModelFactory = Callable[[str], Any]
AudioLoader = Callable[[Path], Any]

SAMPLE_RATE = 16_000
# NeMo recommends 5–25 s per forward pass. A full episode OOMs on 8 GB.
WINDOW_SECONDS = 20.0
OVERLAP_SECONDS = 1.0
MIN_WINDOW_SECONDS = 5.0


class ParakeetBackend:
    """English NVIDIA Parakeet TDT adapter with native word timestamps."""

    DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"

    def __init__(
        self,
        settings: AsrSettings,
        *,
        model_factory: ModelFactory | None = None,
        audio_loader: AudioLoader = load_audio_mono,
        window_seconds: float = WINDOW_SECONDS,
        overlap_seconds: float = OVERLAP_SECONDS,
    ) -> None:
        self._settings = settings
        self._model_factory = model_factory
        self._audio_loader = audio_loader
        self._window_seconds = window_seconds
        self._overlap_seconds = overlap_seconds
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
            logger.info(
                "Parakeet cannot consume ASR prompts or hotwords; "
                "glossary correction still runs after transcription"
            )
        # NeMo's Lhotse loader keeps stereo as (batch, channels, time). Parakeet
        # expects (batch, time); channel_selector="average" is broken in NeMo 3.
        audio = self._audio_loader(media_path)
        duration = len(audio) / SAMPLE_RATE
        words = self._transcribe_windows(
            audio,
            offset=0.0,
            window_seconds=self._window_seconds,
        )
        words.sort(key=lambda word: (word.start, word.end))
        logger.info(
            "asr-parakeet duration=%.1fs words=%d",
            duration,
            len(words),
        )
        _release_cuda()
        return Transcription(words=tuple(words), language="en", duration=duration)

    def close(self) -> None:
        self._model = None
        _release_cuda()

    def _transcribe_windows(
        self,
        audio: Any,
        *,
        offset: float,
        window_seconds: float,
    ) -> list[Word]:
        windows = list(
            _audio_windows(audio, SAMPLE_RATE, window_seconds, self._overlap_seconds)
        )
        words: list[Word] = []
        for index, (local_offset, chunk, is_last) in enumerate(windows):
            try:
                hypothesis = self._load_model().transcribe(
                    [chunk],
                    batch_size=1,
                    timestamps=True,
                )[0]
            except RuntimeError as error:
                if "out of memory" not in str(error).casefold():
                    raise
                _release_cuda()
                smaller = window_seconds / 2
                if smaller < MIN_WINDOW_SECONDS:
                    raise BackendOutOfMemoryError(
                        "Parakeet exhausted VRAM; close other GPU apps"
                    ) from error
                logger.warning(
                    "Parakeet OOM on %.1fs window; retrying at %.1fs",
                    window_seconds,
                    smaller,
                )
                words.extend(
                    self._transcribe_windows(
                        chunk,
                        offset=offset + local_offset,
                        window_seconds=smaller,
                    )
                )
                continue
            chunk_seconds = len(chunk) / SAMPLE_RATE
            words.extend(
                _keep_window_words(
                    _words_from_hypothesis(hypothesis, offset + local_offset),
                    offset + local_offset,
                    chunk_seconds,
                    self._overlap_seconds,
                    first=index == 0,
                    last=is_last,
                )
            )
        return words

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


def _audio_windows(
    audio: Any,
    sample_rate: int,
    window_seconds: float,
    overlap_seconds: float,
) -> Iterator[tuple[float, Any, bool]]:
    window = max(1, int(window_seconds * sample_rate))
    hop = max(1, int((window_seconds - overlap_seconds) * sample_rate))
    total = len(audio)
    start = 0
    while start < total:
        end = min(start + window, total)
        yield start / sample_rate, audio[start:end], end == total
        if end == total:
            return
        start += hop


def _words_from_hypothesis(hypothesis: Any, offset: float) -> tuple[Word, ...]:
    timestamp = getattr(hypothesis, "timestamp", {}) or {}
    items = timestamp.get("word", ())
    return tuple(
        Word(
            start=max(0.0, offset + float(item["start"])),
            end=max(0.0, offset + float(item["start"]), offset + float(item["end"])),
            text=str(item.get("word") or item.get("char") or ""),
        )
        for item in items
        if item.get("start") is not None
        and item.get("end") is not None
        and str(item.get("word") or item.get("char") or "").strip()
    )


def _keep_window_words(
    words: tuple[Word, ...],
    offset: float,
    chunk_seconds: float,
    overlap_seconds: float,
    *,
    first: bool,
    last: bool,
) -> tuple[Word, ...]:
    start_floor = offset if first else offset + overlap_seconds / 2
    end_ceil = None if last else offset + chunk_seconds - overlap_seconds / 2
    kept: list[Word] = []
    for word in words:
        midpoint = (word.start + word.end) / 2
        if midpoint < start_floor:
            continue
        if end_ceil is not None and midpoint >= end_ceil:
            continue
        kept.append(word)
    return tuple(kept)


def _release_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
