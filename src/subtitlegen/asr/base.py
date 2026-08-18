from pathlib import Path
from typing import Protocol

from subtitlegen.asr.capabilities import BackendCapabilities
from subtitlegen.asr.context import AsrContext
from subtitlegen.domain.models import Transcription


class AsrBackend(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities:
        """Describe backend behavior used for validation and preset selection."""

    def transcribe(
        self,
        media_path: Path,
        *,
        language: str | None = None,
        context: AsrContext | None = None,
    ) -> Transcription:
        """Transcribe media into normalized timestamped words."""

    def close(self) -> None:
        """Release model resources owned by this backend."""
