"""Speech recognition backend abstractions."""

from subtitlegen.asr.base import AsrBackend
from subtitlegen.asr.capabilities import BackendCapabilities
from subtitlegen.asr.context import AsrContext
from subtitlegen.asr.faster_whisper import FasterWhisperBackend
from subtitlegen.asr.mlx_whisper import MlxWhisperBackend
from subtitlegen.asr.parakeet import ParakeetBackend
from subtitlegen.asr.whisperx import WhisperXBackend

__all__ = [
    "AsrBackend",
    "AsrContext",
    "BackendCapabilities",
    "FasterWhisperBackend",
    "MlxWhisperBackend",
    "ParakeetBackend",
    "WhisperXBackend",
]
