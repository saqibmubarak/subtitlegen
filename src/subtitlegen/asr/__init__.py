"""Speech recognition backend abstractions."""

from subtitlegen.asr.base import AsrBackend
from subtitlegen.asr.faster_whisper import FasterWhisperBackend

__all__ = ["AsrBackend", "FasterWhisperBackend"]
