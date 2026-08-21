from pathlib import Path
from typing import Any

import numpy as np
import pytest

from subtitlegen.asr.context import AsrContext
from subtitlegen.asr.mlx_whisper import MlxWhisperBackend
from subtitlegen.errors import BackendOutOfMemoryError
from subtitlegen.settings import AsrSettings


def test_mlx_backend_normalizes_word_timestamps(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    calls: list[dict[str, Any]] = []

    def transcribe(_path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "language": "en",
            "segments": [
                {
                    "words": [
                        {"start": 0.0, "end": 0.4, "word": " Hi", "probability": 0.8},
                        {"start": None, "end": None, "word": " bad"},
                    ]
                }
            ],
        }

    backend = MlxWhisperBackend(
        AsrSettings(model="large-v3-turbo"),
        transcribe_fn=transcribe,
        audio_loader=lambda _path: np.zeros(6_400, dtype=np.float32),
    )
    result = backend.transcribe(media, context=AsrContext(prompt="Luffy"))

    assert result.words[0].text == " Hi"
    assert result.duration == 0.4
    assert calls[0]["word_timestamps"] is True
    assert calls[0]["initial_prompt"] == "Luffy"
    assert calls[0]["condition_on_previous_text"] is False
    assert calls[0]["hallucination_silence_threshold"] == 2.0
    assert calls[0]["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"


def test_mlx_backend_validates_media_and_maps_model_names(tmp_path: Path) -> None:
    backend = MlxWhisperBackend(AsrSettings(), transcribe_fn=lambda *_args, **_kwargs: {})
    with pytest.raises(FileNotFoundError):
        backend.transcribe(tmp_path / "missing.wav")
    assert MlxWhisperBackend._model_repository("org/model") == "org/model"
    assert MlxWhisperBackend._model_repository("tiny") == "mlx-community/whisper-tiny-mlx"


def test_mlx_backend_retains_context_across_windows(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    calls: list[dict[str, Any]] = []

    def transcribe(_audio: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "language": "en",
            "segments": [{"words": [{"start": 0, "end": 1, "word": " Luffy"}]}],
        }

    backend = MlxWhisperBackend(
        AsrSettings(),
        transcribe_fn=transcribe,
        audio_loader=lambda _path: np.zeros(31 * 16_000, dtype=np.float32),
    )
    result = backend.transcribe(media, context=AsrContext(prompt="Luffy"))
    assert calls[0]["initial_prompt"] == "Luffy"
    assert calls[0]["condition_on_previous_text"] is False
    assert [word.start for word in result.words] == [0]


def test_mlx_backend_provides_oom_guidance(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()

    def transcribe(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("out of memory")

    backend = MlxWhisperBackend(
        AsrSettings(),
        transcribe_fn=transcribe,
        audio_loader=lambda _path: np.zeros(1, dtype=np.float32),
    )
    with pytest.raises(BackendOutOfMemoryError, match="fast preset"):
        backend.transcribe(media)
