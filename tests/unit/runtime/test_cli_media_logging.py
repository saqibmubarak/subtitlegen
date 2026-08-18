import json
import logging
import wave
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

import subtitlegen.cli as cli_module
from subtitlegen.cli import app
from subtitlegen.logging import JsonFormatter
from subtitlegen.media import discover_media, load_audio_mono, media_duration
from subtitlegen.runtime.service import RuntimeResult


class FakeService:
    def __init__(self) -> None:
        self.paths: list[Path] = []
        self.outputs: list[Path] = []
        self.options: list[dict[str, Any]] = []

    def process(self, media_path: Path, output_path: Path, **kwargs: Any) -> RuntimeResult:
        self.paths.append(media_path)
        self.outputs.append(output_path)
        self.options.append(kwargs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
            encoding="utf-8",
        )
        return RuntimeResult("generated", output_path, "job")


def test_discover_media_handles_recursive_paths_and_spaces(tmp_path: Path) -> None:
    video = tmp_path / "folder with spaces" / "Clip.MP4"
    video.parent.mkdir()
    video.touch()
    (video.parent / "ignore.txt").touch()
    assert discover_media(tmp_path, (".mp4",)) == [video.resolve()]
    assert discover_media(video, (".mp4",)) == [video.resolve()]


def test_media_duration_uses_pyav_without_ffmpeg_binary(tmp_path: Path) -> None:
    audio = tmp_path / "second.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\0\0" * 8_000)
    assert 0.9 <= media_duration(audio) <= 1.1
    decoded = load_audio_mono(audio)
    assert 15_900 <= decoded.size <= 16_100


def test_json_formatter_emits_structured_message() -> None:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello world"


def test_cli_validate_and_generate(monkeypatch: Any, tmp_path: Path) -> None:
    runner = CliRunner()
    subtitle = tmp_path / "valid.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    assert runner.invoke(app, ["validate", str(subtitle)]).exit_code == 0

    video = tmp_path / "folder with spaces" / "video.mp4"
    video.parent.mkdir()
    video.touch()
    duplicate = tmp_path / "another folder" / "video.mp4"
    duplicate.parent.mkdir()
    duplicate.touch()
    fake = FakeService()
    monkeypatch.setattr(cli_module, "_service", lambda *_args: (fake, "fake"))
    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        ["generate", str(tmp_path), "--output-dir", str(output_dir)],
    )
    assert result.exit_code == 0
    assert fake.paths == sorted([video.resolve(), duplicate.resolve()])
    assert len(set(fake.outputs)) == 2
    assert output_dir / "folder with spaces" / "video.srt" in fake.outputs
    assert output_dir / "another folder" / "video.srt" in fake.outputs


def test_cli_benchmark_outputs_json(monkeypatch: Any, tmp_path: Path) -> None:
    runner = CliRunner()
    media = tmp_path / "clip.wav"
    media.touch()
    fake = FakeService()
    monkeypatch.setattr(cli_module, "_service", lambda *_args: (fake, "fake"))
    monkeypatch.setattr(cli_module, "media_duration", lambda _path: 10.0)
    result = runner.invoke(
        app,
        ["benchmark", str(media), "--cache-dir", str(tmp_path / "cache")],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["backend"] == "fake"
    assert fake.options[0]["refresh_stages"] is True
