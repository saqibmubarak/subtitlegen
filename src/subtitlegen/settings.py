from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path

from subtitlegen.cues.rules import CueRules


@dataclass(frozen=True, slots=True)
class VadSettings:
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 200
    max_speech_duration_s: float = 30.0

    def __post_init__(self) -> None:
        if self.min_silence_duration_ms < 0 or self.speech_pad_ms < 0:
            raise ValueError("VAD millisecond settings must be non-negative")
        if self.max_speech_duration_s <= 0:
            raise ValueError("maximum speech duration must be positive")


@dataclass(frozen=True, slots=True)
class AsrSettings:
    model: str = "large-v3-turbo"
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = "en"
    beam_size: int = 5
    vad: VadSettings = field(default_factory=VadSettings)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("ASR model must not be empty")
        if self.beam_size <= 0:
            raise ValueError("beam size must be positive")


@dataclass(frozen=True, slots=True)
class AppSettings:
    asr: AsrSettings = field(default_factory=AsrSettings)
    cues: CueRules = field(default_factory=CueRules)
    video_extensions: tuple[str, ...] = (".mp4", ".mkv", ".avi", ".mov", ".wmv")
    parallel_workers: int = 1

    def __post_init__(self) -> None:
        if self.parallel_workers <= 0:
            raise ValueError("parallel workers must be positive")
        if not self.video_extensions:
            raise ValueError("at least one video extension is required")


class SettingsLoader:
    """Load typed application settings from the legacy INI format."""

    def load(self, path: Path | None = None) -> AppSettings:
        parser = configparser.ConfigParser()
        if path is not None and path.exists():
            parser.read(path)

        model_name = parser.get("TRANSCRIPTION", "model_name", fallback="large-v3-turbo")
        model = parser.get("MODELS", model_name, fallback=model_name)
        language_value = parser.get("TRANSCRIPTION", "language", fallback="en").strip()
        language = None if language_value.lower() == "none" else language_value
        extensions = tuple(
            item.strip().lower()
            for item in parser.get(
                "FILES", "video_extensions", fallback=".mp4, .mkv, .avi, .mov, .wmv"
            ).split(",")
            if item.strip()
        )

        vad = VadSettings(
            min_silence_duration_ms=parser.getint(
                "VAD", "min_silence_duration_ms", fallback=500
            ),
            speech_pad_ms=parser.getint("VAD", "speech_pad_ms", fallback=200),
            max_speech_duration_s=parser.getfloat(
                "VAD", "max_speech_duration_s", fallback=30.0
            ),
        )
        cues = CueRules(
            max_duration_seconds=parser.getfloat(
                "CUES", "max_duration_seconds", fallback=6.0
            ),
            max_characters=parser.getint("CUES", "max_characters", fallback=84),
            max_gap_seconds=parser.getfloat("CUES", "max_gap_seconds", fallback=0.9),
            punctuation_flush_min_seconds=parser.getfloat(
                "CUES", "punctuation_flush_min_seconds", fallback=1.5
            ),
        )
        return AppSettings(
            asr=AsrSettings(
                model=model,
                device=parser.get("TRANSCRIPTION", "device", fallback="auto").lower(),
                compute_type=parser.get(
                    "TRANSCRIPTION", "compute_type", fallback="auto"
                ).lower(),
                language=language,
                beam_size=parser.getint("TRANSCRIPTION", "beam_size", fallback=5),
                vad=vad,
            ),
            cues=cues,
            video_extensions=extensions,
            parallel_workers=parser.getint(
                "TRANSCRIPTION", "parallel_workers", fallback=1
            ),
        )
