from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from subtitlegen.domain.models import Cue
from subtitlegen.errors import EmptySubtitleError


def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("timestamp must be non-negative")
    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


class SrtWriter:
    def write(self, cues: Iterable[Cue], output_path: Path) -> None:
        cue_list = list(cues)
        if not cue_list:
            raise EmptySubtitleError("refusing to write an empty SRT file")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        blocks = [
            (
                f"{index}\n"
                f"{format_srt_timestamp(cue.start)} --> {format_srt_timestamp(cue.end)}\n"
                f"{cue.text.strip()}\n"
            )
            for index, cue in enumerate(cue_list, start=1)
        ]
        output_path.write_text("\n".join(blocks), encoding="utf-8")
