from pathlib import Path

import pytest

from subtitlegen.cues.builder import CueBuilder
from subtitlegen.domain.models import Cue, Transcription, Word
from subtitlegen.errors import EmptySubtitleError
from subtitlegen.export.srt import SrtWriter, format_srt_timestamp
from subtitlegen.pipeline import SubtitleGenerator


class FakeBackend:
    def transcribe(self, _media_path: Path, *, language: str | None = None) -> Transcription:
        return Transcription(
            (Word(0, 0.5, "Hello"), Word(0.5, 1.2, " world.")),
            language or "en",
            1.2,
        )


class EmptyBackend:
    def transcribe(self, _media_path: Path, *, language: str | None = None) -> Transcription:
        return Transcription((), language or "en", 0)


def test_srt_timestamp_and_writer(tmp_path: Path) -> None:
    assert format_srt_timestamp(3661.234) == "01:01:01,234"
    with pytest.raises(ValueError):
        format_srt_timestamp(-1)
    output = tmp_path / "nested" / "output.srt"
    SrtWriter().write([Cue(0, 1.234, "Hello")], output)
    assert output.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,234\nHello\n"
    )
    with pytest.raises(EmptySubtitleError):
        SrtWriter().write([], tmp_path / "empty.srt")


def test_subtitle_generator_coordinates_backend_builder_and_writer(tmp_path: Path) -> None:
    output = tmp_path / "output.srt"
    generator = SubtitleGenerator(FakeBackend(), CueBuilder(), SrtWriter())

    cues = generator.generate(tmp_path / "input.mp4", output, language="en")

    assert [cue.text for cue in cues] == ["Hello world."]
    assert output.exists()


def test_subtitle_generator_rejects_empty_transcription(tmp_path: Path) -> None:
    generator = SubtitleGenerator(EmptyBackend(), CueBuilder(), SrtWriter())
    with pytest.raises(EmptySubtitleError):
        generator.generate(tmp_path / "input.mp4", tmp_path / "output.srt")
