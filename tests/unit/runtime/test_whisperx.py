from pathlib import Path
from typing import Any

import pytest

from subtitlegen.asr.context import AsrContext
from subtitlegen.asr.whisperx import WhisperXBackend
from subtitlegen.errors import BackendOutOfMemoryError, BackendUnavailableError
from subtitlegen.settings import AsrSettings


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, _audio: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"language": "en", "segments": [{"text": "Buggy"}]}


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


def test_whisperx_aligns_words_reuses_models_and_passes_context(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    api = FakeWhisperX()
    backend = WhisperXBackend(
        AsrSettings(device="cuda", whisperx_batch_size=3),
        api=api,
    )

    first = backend.transcribe(media, context=AsrContext("Buggy", ("Buggy",)))
    backend.transcribe(media)

    assert first.words[0].text == " Buggy."
    assert first.duration == 1
    assert api.align_loads == 1
    assert api.model.calls[0]["batch_size"] == 3
    assert api.load_calls == [
        {
            "compute_type": "float16",
            "asr_options": {"initial_prompt": "Buggy", "hotwords": "Buggy"},
        }
    ]
    assert backend.capabilities.requires_cuda
    backend.close()
    backend.transcribe(media)
    assert api.align_loads == 2
    assert len(api.load_calls) == 2


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
