from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import median

from subtitlegen.domain.models import Cue

_TIMING = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3}) --> (?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)


@dataclass(frozen=True, slots=True)
class TimingReport:
    cue_count: int
    median_duration: float
    max_duration: float
    cues_over_limit: int
    overlaps: int


def parse_srt(path: Path) -> list[Cue]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    cues: list[Cue] = []
    for expected_index, block in enumerate(re.split(r"\r?\n\r?\n", content), start=1):
        lines = block.splitlines()
        if len(lines) < 3 or lines[0].strip() != str(expected_index):
            raise ValueError(f"invalid SRT block {expected_index}")
        match = _TIMING.fullmatch(lines[1].strip())
        if match is None:
            raise ValueError(f"invalid timing in SRT block {expected_index}")
        text = "\n".join(lines[2:]).strip()
        if not text:
            raise ValueError(f"empty text in SRT block {expected_index}")
        cues.append(
            Cue(
                start=_parse_timestamp(match.group("start")),
                end=_parse_timestamp(match.group("end")),
                text=text,
            )
        )
    return cues


def is_valid_srt(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        cues = parse_srt(path)
    except (OSError, UnicodeError, ValueError):
        return False
    return bool(cues) and all(cue.end > cue.start for cue in cues)


def analyze_cues(cues: list[Cue], *, duration_limit: float = 8.0) -> TimingReport:
    durations = [cue.duration for cue in cues]
    overlaps = sum(cue.start < previous.end for previous, cue in pairwise(cues))
    return TimingReport(
        cue_count=len(cues),
        median_duration=median(durations) if durations else 0.0,
        max_duration=max(durations, default=0.0),
        cues_over_limit=sum(duration > duration_limit for duration in durations),
        overlaps=overlaps,
    )


def _parse_timestamp(value: str) -> float:
    hours, minutes, remainder = value.split(":")
    seconds, milliseconds = remainder.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000
    )
