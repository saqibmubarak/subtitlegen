from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
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
# NeMo recommends 5–25 s per forward pass. A full episode as one tensor OOMs;
# several equal windows in one batch keep the GPU busy.
WINDOW_SECONDS = 20.0
OVERLAP_SECONDS = 1.0
MIN_WINDOW_SECONDS = 5.0
BATCH_SIZE = 8
PREFETCH_LIMIT = 2


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
        batch_size: int = BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Parakeet batch size must be at least 1")
        self._settings = settings
        self._model_factory = model_factory
        self._audio_loader = audio_loader
        self._window_seconds = window_seconds
        self._overlap_seconds = overlap_seconds
        self._batch_size = batch_size
        self._model: Any | None = None
        self._prefetch_pool: ThreadPoolExecutor | None = None
        self._prefetches: dict[Path, Future[Any]] = {}
        self._logged_context = False

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
        if context is not None and not self._logged_context:
            logger.info(
                "Parakeet cannot consume ASR prompts or hotwords; "
                "glossary correction still runs after transcription"
            )
            self._logged_context = True
        # NeMo's Lhotse loader keeps stereo as (batch, channels, time). Parakeet
        # expects (batch, time); channel_selector="average" is broken in NeMo 3.
        audio = self._consume_audio(media_path)
        duration = len(audio) / SAMPLE_RATE
        try:
            words = self._transcribe_windows(
                audio,
                offset=0.0,
                window_seconds=self._window_seconds,
            )
        except Exception:
            # A CUDA fault poisons the process; drop the model so the next
            # file can reload instead of failing on the first tensor move.
            self.close()
            raise
        del audio
        words.sort(key=lambda word: (word.start, word.end))
        logger.info(
            "asr-parakeet duration=%.1fs words=%d",
            duration,
            len(words),
        )
        return Transcription(words=tuple(words), language="en", duration=duration)

    def prefetch_audio(self, media_path: Path) -> None:
        path = media_path.resolve()
        if path in self._prefetches or len(self._prefetches) >= PREFETCH_LIMIT:
            return
        if self._prefetch_pool is None:
            self._prefetch_pool = ThreadPoolExecutor(
                max_workers=PREFETCH_LIMIT,
                thread_name_prefix="parakeet-audio",
            )
        self._prefetches[path] = self._prefetch_pool.submit(self._audio_loader, path)

    def close(self) -> None:
        for future in self._prefetches.values():
            future.cancel()
        self._prefetches.clear()
        self._model = None
        _release_cuda()

    def _consume_audio(self, media_path: Path) -> Any:
        path = media_path.resolve()
        pending = self._prefetches.pop(path, None)
        if pending is not None:
            return pending.result()
        return self._audio_loader(path)

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
        if not windows:
            return []
        try:
            hypotheses = self._decode_batch(
                [chunk for _offset, chunk, _last in windows],
                allow_cuda_retry=True,
            )
        except Exception as error:
            if not _is_oom(error):
                raise
            self.close()
            local_offset, chunk, _is_last = windows[0]
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
            return self._transcribe_windows(
                chunk,
                offset=offset + local_offset,
                window_seconds=smaller,
            )
        words: list[Word] = []
        for index, ((local_offset, chunk, is_last), hypothesis) in enumerate(
            zip(windows, hypotheses, strict=True)
        ):
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

    def _decode_batch(self, chunks: list[Any], *, allow_cuda_retry: bool) -> list[Any]:
        try:
            return self._load_model().transcribe(
                chunks,
                batch_size=min(self._batch_size, len(chunks)),
                timestamps=True,
                verbose=False,
            )
        except Exception as error:
            if _is_oom(error):
                self.close()
                if len(chunks) == 1:
                    raise
                logger.warning(
                    "Parakeet OOM on batch of %d; splitting",
                    len(chunks),
                )
                mid = len(chunks) // 2
                return self._decode_batch(
                    chunks[:mid], allow_cuda_retry=True
                ) + self._decode_batch(chunks[mid:], allow_cuda_retry=True)
            if not _is_cuda_fault(error):
                raise
            self.close()
            if allow_cuda_retry:
                logger.warning("Parakeet CUDA fault; reloading model and retrying batch")
                return self._decode_batch(chunks, allow_cuda_retry=False)
            if len(chunks) > 1:
                logger.warning(
                    "Parakeet CUDA fault on batch of %d; splitting",
                    len(chunks),
                )
                mid = len(chunks) // 2
                return self._decode_batch(
                    chunks[:mid], allow_cuda_retry=True
                ) + self._decode_batch(chunks[mid:], allow_cuda_retry=True)
            raise

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


def _is_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).casefold()


def _is_cuda_fault(error: BaseException) -> bool:
    text = str(error).casefold()
    name = type(error).__name__.casefold()
    return (
        _is_oom(error)
        or "illegal memory access" in text
        or "acceleratorerror" in name
        or "cudaerror" in name
    )


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()
