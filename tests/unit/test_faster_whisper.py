from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from subtitlegen.asr.context import AsrContext
from subtitlegen.asr.faster_whisper import FasterWhisperBackend
from subtitlegen.settings import AsrSettings


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, _path: str, **kwargs: Any) -> tuple[list[Any], Any]:
        self.calls.append(kwargs)
        words = [
            SimpleNamespace(start=0.0, end=0.4, word=" Hello", probability=0.95),
            SimpleNamespace(start=0.4, end=0.8, word=" world.", probability=0.9),
        ]
        return [SimpleNamespace(words=words)], SimpleNamespace(language="en", duration=1.0)


def test_backend_normalizes_words_and_passes_sync_options(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    fake = FakeModel()
    factory_calls: list[dict[str, Any]] = []

    def factory(_model: str, **kwargs: Any) -> FakeModel:
        factory_calls.append(kwargs)
        return fake

    backend = FasterWhisperBackend(
        AsrSettings(model="tiny", device="cpu", compute_type="int8"),
        model_factory=factory,
    )
    result = backend.transcribe(
        media,
        context=AsrContext(prompt="Aang", hotwords=("Aang", "airbender")),
    )

    assert [word.text for word in result.words] == [" Hello", " world."]
    assert factory_calls == [{"device": "cpu", "compute_type": "int8"}]
    call = fake.calls[0]
    assert call["word_timestamps"] is True
    assert call["condition_on_previous_text"] is False
    assert call["initial_prompt"] == "Aang"
    assert call["hotwords"] == "Aang airbender"
    assert call["vad_parameters"]["min_silence_duration_ms"] == 500


def test_backend_reuses_and_can_release_model(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    created: list[FakeModel] = []

    def factory(_model: str, **_kwargs: Any) -> FakeModel:
        created.append(FakeModel())
        return created[-1]

    backend = FasterWhisperBackend(AsrSettings(device="cpu"), model_factory=factory)
    backend.transcribe(media)
    backend.transcribe(media)
    assert len(created) == 1
    backend.close()
    backend.transcribe(media)
    assert len(created) == 2


def test_backend_rejects_missing_media(tmp_path: Path) -> None:
    backend = FasterWhisperBackend(AsrSettings(device="cpu"), model_factory=FakeModel)
    with pytest.raises(FileNotFoundError):
        backend.transcribe(tmp_path / "missing.mp4")
