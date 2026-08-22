from pathlib import Path
from typing import Any

import pytest

from subtitlegen.asr.context import AsrContext
from subtitlegen.asr.whisperx import WhisperXBackend
from subtitlegen.domain.models import Transcription, Word
from subtitlegen.errors import BackendOutOfMemoryError, BackendUnavailableError
from subtitlegen.settings import AsrSettings

_OVERFLOW = "No position encodings are defined for positions >= 448, but got position 449"


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, _audio: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"language": "en", "segments": [{"text": "Buggy"}]}


class OverflowOnceModel:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError(_OVERFLOW)
        return {"language": "en", "segments": [{"text": "ok"}]}


class AlwaysOverflowModel:
    def transcribe(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(_OVERFLOW)


class FakeWhisperX:
    def __init__(self, model: Any | None = None) -> None:
        self.model = model or FakeModel()
        self.load_calls: list[dict[str, Any]] = []
        self.align_loads = 0

    def load_model(self, _name: str, _device: str, **kwargs: Any) -> Any:
        self.load_calls.append(kwargs)
        return self.model

    def load_audio(self, _path: str) -> list[float]:
        return []

    def load_align_model(self, **_kwargs: Any) -> tuple[object, object]:
        self.align_loads += 1
        return object(), object()

    def align(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "word_segments": [
                {"start": 0.5, "end": 1.0, "word": " Buggy.", "score": 0.8},
                {"word": "missing"},
            ]
        }


class FallbackBackend:
    def transcribe(
        self,
        _media_path: Path,
        *,
        language: str | None = None,
        context: AsrContext | None = None,
    ) -> Transcription:
        del language, context
        return Transcription((Word(0.0, 1.0, "fallback"),), "en", 1.0)

    def close(self) -> None:
        return None


def test_whisperx_aligns_words_reuses_models_and_passes_context(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    api = FakeWhisperX()
    backend = WhisperXBackend(
        AsrSettings(device="cuda", whisperx_batch_size=3),
        api=api,
    )

    first = backend.transcribe(media, context=AsrContext("Buggy", ("Buggy",)))
    backend.transcribe(media, context=AsrContext("Buggy", ("Buggy",)))

    assert first.words[0].text == " Buggy."
    assert first.duration == 1
    assert api.align_loads == 1
    assert api.model.calls[0]["batch_size"] == 3
    assert api.load_calls == [
        {
            "compute_type": "float16",
            "asr_options": {
                "initial_prompt": "Buggy",
                "hotwords": "Buggy",
                "max_new_tokens": 224,
            },
        }
    ]
    assert backend.capabilities.requires_cuda
    backend.close()
    backend.transcribe(media)
    assert api.align_loads == 2
    assert len(api.load_calls) == 2
    assert api.load_calls[1]["asr_options"]["initial_prompt"] is None


def test_whisperx_retries_without_prompt_after_decoder_overflow(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    api = FakeWhisperX(OverflowOnceModel())
    backend = WhisperXBackend(AsrSettings(device="cuda"), api=api)

    result = backend.transcribe(media, context=AsrContext("Buggy", ("Buggy",)))

    assert result.words[0].text == " Buggy."
    assert len(api.load_calls) == 2
    assert api.load_calls[0]["asr_options"]["initial_prompt"] == "Buggy"
    assert api.load_calls[1]["asr_options"]["initial_prompt"] is None


def test_whisperx_falls_back_when_decoder_still_overflows(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    backend = WhisperXBackend(
        AsrSettings(device="cuda"),
        api=FakeWhisperX(AlwaysOverflowModel()),
        fallback=FallbackBackend(),
    )

    result = backend.transcribe(media, context=AsrContext("Buggy", ("Buggy",)))

    assert result.words[0].text == "fallback"


def test_whisperx_validates_device_batch_and_oom(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    with pytest.raises(ValueError):
        WhisperXBackend(AsrSettings(), batch_size=0)
    with pytest.raises(BackendUnavailableError):
        WhisperXBackend(AsrSettings(device="cpu"), api=FakeWhisperX()).transcribe(media)

    class OomModel:
        def transcribe(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(BackendOutOfMemoryError, match="whisperx_batch_size"):
        WhisperXBackend(
            AsrSettings(device="cuda"),
            api=FakeWhisperX(OomModel()),
        ).transcribe(media)


def test_whisperx_rejects_missing_media(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        WhisperXBackend(AsrSettings(device="cuda"), api=FakeWhisperX()).transcribe(
            tmp_path / "missing.wav"
        )
