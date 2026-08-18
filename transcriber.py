"""Compatibility wrapper around the normalized faster-whisper adapter."""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent / "src"))

from subtitlegen.asr.faster_whisper import FasterWhisperBackend
from subtitlegen.settings import AsrSettings


def transcribe_video(
    video_path: Path,
    model_identifier: str,
    device: str,
    language: str | None,
    compute_type: str,
) -> dict[str, Any]:
    result = FasterWhisperBackend(
        AsrSettings(
            model=model_identifier,
            device=device,
            language=language,
            compute_type=compute_type,
        )
    ).transcribe(video_path, language=language)
    return {
        "words": [
            {
                "start": word.start,
                "end": word.end,
                "text": word.text,
                "probability": word.probability,
            }
            for word in result.words
        ],
        "language": result.language,
        "duration": result.duration,
    }