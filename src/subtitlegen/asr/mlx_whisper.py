from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from subtitlegen.domain.models import Transcription, Word
from subtitlegen.media import load_audio_mono
from subtitlegen.settings import AsrSettings

MlxTranscribe = Callable[..., dict[str, Any]]
AudioLoader = Callable[[Path], Any]


class MlxWhisperBackend:
    """Apple Silicon Whisper adapter using mlx-whisper."""

    def __init__(
        self,
        settings: AsrSettings,
        *,
        transcribe_fn: MlxTranscribe | None = None,
        audio_loader: AudioLoader = load_audio_mono,
    ) -> None:
        self._settings = settings
        self._transcribe_fn = transcribe_fn
        self._audio_loader = audio_loader

    def transcribe(self, media_path: Path, *, language: str | None = None) -> Transcription:
        if not media_path.exists():
            raise FileNotFoundError(media_path)
        transcribe = self._transcribe_fn
        if transcribe is None:
            try:
                import mlx_whisper
            except ImportError as error:
                raise RuntimeError(
                    "MLX backend is unavailable; install subtitlegen[mac]"
                ) from error
            transcribe = mlx_whisper.transcribe

        result = transcribe(
            self._audio_loader(media_path),
            path_or_hf_repo=self._model_repository(self._settings.model),
            language=language if language is not None else self._settings.language,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        words: list[Word] = []
        for segment in result.get("segments", []):
            for item in segment.get("words", []):
                start = item.get("start")
                end = item.get("end")
                text = str(item.get("word", ""))
                if start is None or end is None or not text.strip():
                    continue
                words.append(
                    Word(
                        start=max(0.0, float(start)),
                        end=max(float(start), float(end)),
                        text=text,
                        probability=item.get("probability"),
                    )
                )
        words.sort(key=lambda word: (word.start, word.end))
        duration = max((word.end for word in words), default=0.0)
        return Transcription(
            words=tuple(words),
            language=str(result.get("language") or language or "unknown"),
            duration=duration,
        )

    @staticmethod
    def _model_repository(model: str) -> str:
        if "/" in model:
            return model
        aliases = {
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "turbo": "mlx-community/whisper-large-v3-turbo",
        }
        return aliases.get(model, f"mlx-community/whisper-{model}-mlx")
