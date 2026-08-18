from pathlib import Path
from typing import Protocol

from subtitlegen.asr.context import AsrContext
from subtitlegen.domain.models import Transcription


class AsrBackend(Protocol):
    def transcribe(
        self,
        media_path: Path,
        *,
        language: str | None = None,
        context: AsrContext | None = None,
    ) -> Transcription:
        """Transcribe media into normalized timestamped words."""
