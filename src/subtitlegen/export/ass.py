from __future__ import annotations

from pathlib import Path

from subtitlegen.visual.models import StyledCue

HEADER = "\n".join(
    (
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
            "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
            "MarginR, MarginV, Encoding"
        ),
        (
            "Style: Dialogue,Arial,48,&H00FFFFFF,&H000000FF,&H00101010,&H80000000,"
            "0,0,0,0,100,100,0,0,1,3,1,2,60,60,42,1"
        ),
        (
            "Style: OnScreen,Arial,44,&H0000FFFF,&H000000FF,&H00101010,&H80000000,"
            "-1,0,0,0,100,100,0,0,1,3,1,8,60,60,42,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        "",
    )
)


class AssWriter:
    def render(self, cues: list[StyledCue] | tuple[StyledCue, ...]) -> str:
        lines = [HEADER.rstrip()]
        lines.extend(
            "Dialogue: 0,"
            f"{_format_time(cue.start)},{_format_time(cue.end)},"
            f"{cue.style},,0,0,0,,{_escape(cue.text)}"
            for cue in cues
        )
        return "\n".join(lines) + "\n"

    def write(
        self,
        cues: list[StyledCue] | tuple[StyledCue, ...],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.render(cues), encoding="utf-8")


def parse_ass_events(content: str) -> tuple[StyledCue, ...]:
    events: list[StyledCue] = []
    for line in content.splitlines():
        if not line.startswith("Dialogue: "):
            continue
        fields = line.removeprefix("Dialogue: ").split(",", 9)
        if len(fields) != 10:
            raise ValueError("malformed ASS dialogue event")
        events.append(
            StyledCue(
                _parse_time(fields[1]),
                _parse_time(fields[2]),
                _unescape(fields[9]),
                fields[3],
            )
        )
    return tuple(events)


def _format_time(seconds: float) -> str:
    centiseconds = round(seconds * 100)
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _parse_time(value: str) -> float:
    hours, minutes, remainder = value.split(":")
    seconds, centiseconds = remainder.split(".")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(centiseconds) / 100


def _escape(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def _unescape(text: str) -> str:
    output: list[str] = []
    index = 0
    replacements = {r"\\": "\\", r"\N": "\n", r"\{": "{", r"\}": "}"}
    while index < len(text):
        pair = text[index : index + 2]
        if pair in replacements:
            output.append(replacements[pair])
            index += 2
        else:
            output.append(text[index])
            index += 1
    return "".join(output)
