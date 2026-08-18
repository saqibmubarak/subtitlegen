from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Compatibility for `python main.py` before the package is installed.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from subtitlegen.asr.faster_whisper import FasterWhisperBackend
from subtitlegen.cues.builder import CueBuilder
from subtitlegen.export.srt import SrtWriter
from subtitlegen.pipeline import SubtitleGenerator
from subtitlegen.settings import SettingsLoader
from subtitlegen.validation import is_valid_srt


def find_video_files(input_path: Path, extensions: tuple[str, ...]) -> list[Path]:
    if input_path.is_file():
        return [input_path.resolve()] if input_path.suffix.lower() in extensions else []
    return sorted(
        path.resolve()
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synchronized SRT subtitles.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.ini"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = args.input_path.resolve()
    if not input_path.exists():
        parser.error(f"input path not found: {input_path}")

    settings = SettingsLoader().load(args.config)
    generator = SubtitleGenerator(
        FasterWhisperBackend(settings.asr),
        CueBuilder(settings.cues),
        SrtWriter(),
    )
    videos = find_video_files(input_path, settings.video_extensions)
    if not videos:
        print(f"No supported video files found in {input_path}")
        return 0

    failed = 0
    for video in videos:
        output = video.with_suffix(".srt")
        if is_valid_srt(output) and not args.overwrite:
            print(f"Skipping existing subtitle: {output.name}")
            continue
        try:
            cues = generator.generate(video, output, language=settings.asr.language)
            print(f"Created {output.name} with {len(cues)} cues")
        except Exception as error:
            failed += 1
            print(f"Failed {video.name}: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
