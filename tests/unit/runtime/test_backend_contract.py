from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from subtitlegen.asr.base import AsrBackend
from subtitlegen.asr.faster_whisper import FasterWhisperBackend
from subtitlegen.asr.mlx_whisper import MlxWhisperBackend
from subtitlegen.asr.parakeet import ParakeetBackend
from subtitlegen.asr.whisperx import WhisperXBackend
from subtitlegen.settings import AsrSettings

BackendBuilder = Callable[[], AsrBackend]


class FasterModel:
    def transcribe(self, *_args: Any, **_kwargs: Any) -> tuple[list[Any], Any]:
        words = [SimpleNamespace(start=-0.5, end=-0.1, word=" word", probability=0.9)]
        return [SimpleNamespace(words=words)], SimpleNamespace(language="en", duration=1)


class WhisperXApi:
    def load_model(self, *_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            transcribe=lambda *_args, **_kwargs: {"language": "en", "segments": [{}]}
        )

    def load_audio(self, _path: str) -> list[float]:
        return []

    def load_align_model(self, **_kwargs: Any) -> tuple[object, object]:
        return object(), object()

    def align(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "word_segments": [
                {"start": -0.5, "end": -0.1, "word": " word", "score": 0.9}
            ]
        }


class ParakeetModel:
    def transcribe(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return [
            SimpleNamespace(
                timestamp={"word": [{"start": -0.5, "end": -0.1, "word": " word"}]}
            )
        ]


def _mlx_result(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "language": "en",
        "segments": [
            {
                "words": [
                    {"start": -0.5, "end": -0.1, "word": " word", "probability": 0.9}
                ]
            }
        ],
    }


@pytest.mark.parametrize(
    "build",
    [
        lambda: FasterWhisperBackend(
            AsrSettings(device="cpu"),
            model_factory=lambda *_args, **_kwargs: FasterModel(),
        ),
        lambda: MlxWhisperBackend(
            AsrSettings(),
            transcribe_fn=_mlx_result,
            audio_loader=lambda _path: np.zeros(16_000, dtype=np.float32),
        ),
        lambda: WhisperXBackend(AsrSettings(device="cuda"), api=WhisperXApi()),
        lambda: ParakeetBackend(
            AsrSettings(),
            model_factory=lambda _name: ParakeetModel(),
            audio_loader=lambda _path: np.zeros(16_000, dtype=np.float32),
        ),
    ],
    ids=["faster-whisper", "mlx", "whisperx", "parakeet"],
)
def test_backend_contract(build: BackendBuilder, tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    backend = build()

    result = backend.transcribe(media, language="en")

    assert result.language == "en"
    assert result.words
    assert all(word.start >= 0 and word.end >= word.start for word in result.words)
    assert result.words[0].start == result.words[0].end == 0
    assert list(result.words) == sorted(result.words, key=lambda word: (word.start, word.end))
    assert backend.capabilities.word_timestamps
    backend.close()
