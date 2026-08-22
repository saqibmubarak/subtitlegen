import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from subtitlegen.asr.context import AsrContext
from subtitlegen.asr.parakeet import ParakeetBackend
from subtitlegen.errors import BackendOutOfMemoryError
from subtitlegen.settings import AsrSettings

SILENCE = np.zeros(16_000, dtype=np.float32)


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: list[Any], **kwargs: Any) -> list[Any]:
        self.calls.append({"audio": audio, **kwargs})
        return [
            SimpleNamespace(
                timestamp={
                    "word": [
                        {"start": 1.0, "end": 1.4, "word": " world"},
                        {"start": 0.0, "end": 0.5, "word": "Hello"},
                        {"start": None, "end": None, "word": "skip"},
                    ]
                }
            )
        ]


def test_parakeet_normalizes_timestamps_reuses_and_releases_model(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    created: list[FakeModel] = []
    names: list[str] = []

    def factory(name: str) -> FakeModel:
        names.append(name)
        created.append(FakeModel())
        return created[-1]

    backend = ParakeetBackend(
        AsrSettings(model="large-v3-turbo"),
        model_factory=factory,
        audio_loader=lambda _path: SILENCE,
    )
    result = backend.transcribe(media)
    backend.transcribe(media)

    assert [word.text for word in result.words] == ["Hello", " world"]
    assert result.language == "en"
    assert result.duration == 1.0
    assert names == [ParakeetBackend.DEFAULT_MODEL, ParakeetBackend.DEFAULT_MODEL]
    assert created[0].calls[0]["batch_size"] == 1
    assert created[0].calls[0]["timestamps"] is True
    assert created[0].calls[0]["verbose"] is False
    assert np.array_equal(created[0].calls[0]["audio"][0], SILENCE)
    assert not backend.capabilities.context_prompt
    assert len(created) == 2

    backend.transcribe(media)
    assert len(created) == 3


def test_parakeet_ignores_decoder_context_and_rejects_non_english(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    backend = ParakeetBackend(
        AsrSettings(),
        model_factory=lambda _name: FakeModel(),
        audio_loader=lambda _path: SILENCE,
    )
    with pytest.raises(ValueError, match="English"):
        backend.transcribe(media, language="ja")
    result = backend.transcribe(media, context=AsrContext("Buggy"))
    assert [word.text for word in result.words] == ["Hello", " world"]
    with pytest.raises(FileNotFoundError):
        backend.transcribe(tmp_path / "missing.wav")

    class OomModel:
        def transcribe(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(BackendOutOfMemoryError, match="close other GPU"):
        ParakeetBackend(
            AsrSettings(),
            model_factory=lambda _name: OomModel(),
            audio_loader=lambda _path: SILENCE,
        ).transcribe(media)


def test_parakeet_downmixes_stereo_media_before_nemo(tmp_path: Path) -> None:
    media = tmp_path / "stereo.wav"
    with wave.open(str(media), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\0\0" * 8_000 * 2)

    model = FakeModel()
    ParakeetBackend(AsrSettings(), model_factory=lambda _name: model).transcribe(media)

    audio = model.calls[0]["audio"][0]
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert 15_900 <= audio.size <= 16_100


def test_parakeet_windows_long_audio_and_offsets_timestamps(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    audio = np.zeros(45 * 16_000, dtype=np.float32)
    model = FakeModel()
    result = ParakeetBackend(
        AsrSettings(),
        model_factory=lambda _name: model,
        audio_loader=lambda _path: audio,
        window_seconds=20.0,
        overlap_seconds=0.0,
    ).transcribe(media)

    assert len(model.calls) == 3
    assert [len(call["audio"][0]) for call in model.calls] == [
        20 * 16_000,
        20 * 16_000,
        5 * 16_000,
    ]
    assert result.duration == 45.0
    assert {word.start for word in result.words} >= {0.0, 20.0, 40.0}


def test_parakeet_splits_window_after_oom(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    audio = np.zeros(20 * 16_000, dtype=np.float32)

    class SplitThenOk:
        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, chunks: list[Any], **_kwargs: Any) -> list[Any]:
            self.calls += 1
            if len(chunks[0]) > 12 * 16_000:
                raise RuntimeError("CUDA out of memory")
            return FakeModel().transcribe(chunks)

    model = SplitThenOk()
    result = ParakeetBackend(
        AsrSettings(),
        model_factory=lambda _name: model,
        audio_loader=lambda _path: audio,
        window_seconds=20.0,
        overlap_seconds=1.0,
    ).transcribe(media)

    assert model.calls >= 3
    assert result.words
    assert result.duration == 20.0


def test_parakeet_reloads_model_after_cuda_illegal_access(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    loads = {"count": 0}

    class FaultModel:
        def transcribe(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    def factory(_name: str) -> Any:
        loads["count"] += 1
        return FaultModel() if loads["count"] == 1 else FakeModel()

    result = ParakeetBackend(
        AsrSettings(),
        model_factory=factory,
        audio_loader=lambda _path: SILENCE,
    ).transcribe(media)

    assert loads["count"] == 2
    assert result.words
