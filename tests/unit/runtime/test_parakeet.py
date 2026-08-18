from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from subtitlegen.asr.context import AsrContext
from subtitlegen.asr.parakeet import ParakeetBackend
from subtitlegen.errors import BackendOutOfMemoryError, BackendUnavailableError
from subtitlegen.settings import AsrSettings


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, _paths: list[str], **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
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

    backend = ParakeetBackend(AsrSettings(model="large-v3-turbo"), model_factory=factory)
    result = backend.transcribe(media)
    backend.transcribe(media)

    assert [word.text for word in result.words] == ["Hello", " world"]
    assert result.language == "en"
    assert result.duration == 1.4
    assert names == [ParakeetBackend.DEFAULT_MODEL]
    assert created[0].calls[0] == {"batch_size": 1, "timestamps": True}
    assert not backend.capabilities.context_prompt

    backend.close()
    backend.transcribe(media)
    assert len(created) == 2


def test_parakeet_rejects_context_languages_missing_media_and_oom(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    backend = ParakeetBackend(AsrSettings(), model_factory=lambda _name: FakeModel())
    with pytest.raises(ValueError, match="English"):
        backend.transcribe(media, language="ja")
    with pytest.raises(BackendUnavailableError, match="context"):
        backend.transcribe(media, context=AsrContext("Buggy"))
    with pytest.raises(FileNotFoundError):
        backend.transcribe(tmp_path / "missing.wav")

    class OomModel:
        def transcribe(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(BackendOutOfMemoryError, match="close other GPU"):
        ParakeetBackend(AsrSettings(), model_factory=lambda _name: OomModel()).transcribe(
            media
        )
