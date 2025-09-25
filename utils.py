# utils.py

from pathlib import Path
from typing import List, Dict, Any, Optional


def format_timestamp(seconds: float) -> str:
    """Formats a timestamp in seconds to the SRT format (HH:MM:SS,mmm)."""
    if seconds is None:
        return "00:00:00,000"

    milliseconds = int(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(segments: List[Dict[str, Any]], output_path: Path):
    """
    Writes Whisper transcription segments to an SRT file at the specified path.

    Requirement: Output is a .srt file with same base name and path as video.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            # 1. Subtitle Index
            f.write(f"{i}\n")

            # 2. Timecodes
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            f.write(f"{start} --> {end}\n")

            # 3. Text Content
            f.write(f"{segment['text'].strip()}\n\n")


def find_video_files(input_path: Path, extensions: List[str]) -> List[Path]:
    """
    Traverses a directory recursively or checks a single file for supported video extensions.

    Requirement: Batch process, directory traversal.
    """
    video_files = []

    if input_path.is_file():
        if input_path.suffix.lower() in extensions:
            video_files.append(input_path)
    elif input_path.is_dir():
        # Recursive glob search for all files matching extensions
        for ext in extensions:
            video_files.extend(input_path.rglob(f'*{ext}'))

    return [p.resolve() for p in video_files]