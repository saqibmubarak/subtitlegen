from pathlib import Path
from typing import Any

import pytest

from subtitlegen.asr.mlx_whisper import MlxWhisperBackend
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
        audio_loader=lambda _path: [0.0],
    )
    result = backend.transcribe(media)

    assert result.words[0].text == " Hi"
    assert result.duration == 0.4
    assert calls[0]["word_timestamps"] is True
    assert calls[0]["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"


def test_mlx_backend_validates_media_and_maps_model_names(tmp_path: Path) -> None:
    backend = MlxWhisperBackend(AsrSettings(), transcribe_fn=lambda *_args, **_kwargs: {})
    with pytest.raises(FileNotFoundError):
        backend.transcribe(tmp_path / "missing.wav")
    assert MlxWhisperBackend._model_repository("org/model") == "org/model"
    assert MlxWhisperBackend._model_repository("tiny") == "mlx-community/whisper-tiny-mlx"
