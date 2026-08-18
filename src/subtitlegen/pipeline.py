from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from subtitlegen.asr.base import AsrBackend
from subtitlegen.domain.models import Cue, Word
from subtitlegen.errors import EmptySubtitleError


class CueAssembler(Protocol):
    def build(self, words: Iterable[Word]) -> list[Cue]:
        """Build display cues from timestamped words."""


class SubtitleWriter(Protocol):
    def write(self, cues: Iterable[Cue], output_path: Path) -> None:
        """Write cues to a subtitle file."""


class SubtitleGenerator:
    """Coordinate transcription, cue building, and SRT output."""

    def __init__(
        self,
        backend: AsrBackend,
        cue_builder: CueAssembler,
        writer: SubtitleWriter,
    ) -> None:
        self._backend = backend
        self._cue_builder = cue_builder
        self._writer = writer

    def generate(
        self,
        media_path: Path,
        output_path: Path,
        *,
        language: str | None = None,
    ) -> list[Cue]:
        transcription = self._backend.transcribe(media_path, language=language)
        cues = self._cue_builder.build(transcription.words)
        if not cues:
            raise EmptySubtitleError(f"no speech was detected in {media_path}")
        self._writer.write(cues, output_path)
        return cues
