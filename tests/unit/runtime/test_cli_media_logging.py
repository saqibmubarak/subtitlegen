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
from subtitlegen.profiles.models import SeriesProfile
from subtitlegen.profiles.repository import ProfileRepository
from subtitlegen.runtime.capabilities import DeviceCapabilities
from subtitlegen.runtime.service import RuntimeResult
from subtitlegen.settings import AppSettings, AsrSettings
from subtitlegen.visual.service import MultimodalResult


class FakeService:
    def __init__(self) -> None:
        self.paths: list[Path] = []
        self.outputs: list[Path] = []
        self.options: list[dict[str, Any]] = []
        self.closed = False

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

    def close(self) -> None:
        self.closed = True


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
    selected_profiles: list[Any] = []
    selected_backends: list[str] = []

    def fake_service(*args: Any, **_kwargs: Any) -> tuple[FakeService, str]:
        selected_profiles.append(args[3])
        selected_backends.append(args[1])
        return fake, "fake"

    monkeypatch.setattr(cli_module, "_service", fake_service)
    monkeypatch.setattr(
        cli_module.DeviceCapabilities,
        "detect",
        lambda: DeviceCapabilities("Darwin", "arm64", 0, True),
    )
    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "generate",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--profile",
            "avatar",
            "--preset",
            "quality",
            "--no-visual-text",
        ],
    )
    assert result.exit_code == 0
    assert fake.paths == sorted([video.resolve(), duplicate.resolve()])
    assert len(set(fake.outputs)) == 2
    assert output_dir / "folder with spaces" / "video.srt" in fake.outputs
    assert output_dir / "another folder" / "video.srt" in fake.outputs
    assert selected_profiles[0].profile_id == "avatar"
    assert selected_backends == ["mlx"]
    assert fake.closed


def test_cli_benchmark_outputs_json(monkeypatch: Any, tmp_path: Path) -> None:
    runner = CliRunner()
    media = tmp_path / "clip.wav"
    media.touch()
    fake = FakeService()
    monkeypatch.setattr(
        cli_module, "_service", lambda *_args, **_kwargs: (fake, "fake")
    )
    monkeypatch.setattr(cli_module, "media_duration", lambda _path: 10.0)
    result = runner.invoke(
        app,
        ["benchmark", str(media), "--cache-dir", str(tmp_path / "cache")],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["backend"] == "fake"
    assert fake.closed
    assert fake.options[0]["refresh_stages"] is True


def test_cli_benchmark_preserves_interrupt_and_closes_service(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    media = tmp_path / "clip.wav"
    media.touch()

    class InterruptingService(FakeService):
        def process(
            self,
            media_path: Path,
            output_path: Path,
            **kwargs: Any,
        ) -> RuntimeResult:
            raise KeyboardInterrupt

    service = InterruptingService()
    monkeypatch.setattr(
        cli_module,
        "_service",
        lambda *_args, **_kwargs: (service, "fake"),
    )

    result = CliRunner().invoke(app, ["benchmark", str(media)])

    assert result.exit_code == 130
    assert not isinstance(result.exception, NameError)
    assert service.closed


def test_cli_applies_preset_language_before_execution(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        cli_module.DeviceCapabilities,
        "detect",
        lambda: DeviceCapabilities("Linux", "x86_64", 1, False),
    )
    settings, backend = cli_module._resolve_runtime(
        AppSettings(asr=AsrSettings(language="ja")),
        "auto",
        "english-fast",
    )
    assert backend == "parakeet"
    assert settings.asr.language == "en"


def test_cli_visual_text_runs_after_dialogue_and_writes_ass(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    dialogue_service = FakeService()

    class FakeMultimodal:
        def __init__(self) -> None:
            self.outputs: list[Path] = []
            self.closed = False

        def process(
            self,
            _video: Path,
            _dialogue: Path,
            output: Path,
        ) -> MultimodalResult:
            self.outputs.append(output)
            output.write_text("ASS", encoding="utf-8")
            return MultimodalResult(output, 1, 1)

        def close(self) -> None:
            self.closed = True

    multimodal = FakeMultimodal()
    monkeypatch.setattr(
        cli_module,
        "_service",
        lambda *_args, **_kwargs: (dialogue_service, "fake"),
    )
    monkeypatch.setattr(
        cli_module,
        "_visual_service",
        lambda *_args, **_kwargs: multimodal,
    )

    result = CliRunner().invoke(app, ["generate", str(video), "--visual-text"])

    assert result.exit_code == 0
    assert multimodal.outputs == [video.with_suffix(".ass")]
    assert multimodal.closed


def test_cli_auto_profile_uses_shipped_match_and_skips_visual_for_avatar(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    video = tmp_path / "Avatar - S01E01 - The Boy in the Iceberg.mp4"
    video.touch()
    fake = FakeService()
    selected_profiles: list[Any] = []

    def fake_service(*args: Any, **_kwargs: Any) -> tuple[FakeService, str]:
        selected_profiles.append(args[3])
        return fake, "fake"

    monkeypatch.setattr(cli_module, "_service", fake_service)
    monkeypatch.setattr(
        cli_module,
        "_visual_service",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("visual")),
    )

    result = CliRunner().invoke(
        app,
        [
            "generate",
            str(video),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-visual-text",
        ],
    )

    assert result.exit_code == 0, result.output
    assert selected_profiles[0].profile_id == "avatar"
    assert fake.closed


def test_cli_auto_profile_enables_visual_when_resolver_requests_it(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    video = tmp_path / "Made Up Series - S01E02.mp4"
    video.touch()
    fake = FakeService()
    from subtitlegen.profiles.identity import MediaIdentity
    from subtitlegen.profiles.models import GlossaryEntry, SeriesProfile
    from subtitlegen.profiles.resolver import ResolvedProfile

    profile = SeriesProfile(
        1,
        "made-up-series",
        "Made Up Series",
        "en",
        (GlossaryEntry("Hero"),),
    )

    class FakeMultimodal:
        def __init__(self) -> None:
            self.closed = False

        def process(self, _video: Path, _dialogue: Path, output: Path) -> MultimodalResult:
            output.write_text("ASS", encoding="utf-8")
            return MultimodalResult(output, 1, 1)

        def close(self) -> None:
            self.closed = True

    multimodal = FakeMultimodal()
    monkeypatch.setattr(
        cli_module,
        "_resolve_profile",
        lambda *_args, **_kwargs: ResolvedProfile(
            profile,
            MediaIdentity("Made Up Series", "made-up-series", episode="2"),
            "wikipedia",
            True,
        ),
    )
    monkeypatch.setattr(cli_module, "_service", lambda *_args, **_kwargs: (fake, "fake"))
    monkeypatch.setattr(cli_module, "_visual_service", lambda *_args, **_kwargs: multimodal)

    result = CliRunner().invoke(
        app,
        ["generate", str(video), "--cache-dir", str(tmp_path / "cache")],
    )

    assert result.exit_code == 0, result.output
    assert multimodal.closed
    assert video.with_suffix(".ass").is_file()


def test_enrich_profile_adds_repeated_names_and_replaces_processor(tmp_path: Path) -> None:
    output = tmp_path / "out.srt"
    output.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nAang meets Katara\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nAang and Katara leave\n",
        encoding="utf-8",
    )

    class Service:
        def __init__(self) -> None:
            self.processor: Any = None

        def set_cue_processor(self, processor: Any) -> None:
            self.processor = processor

    service = Service()
    updated = cli_module._enrich_profile(
        SeriesProfile(1, "avatar", "Avatar", "en", ()),
        output,
        ProfileRepository(tmp_path / "profiles"),
        service,  # type: ignore[arg-type]
        local_correction=True,
    )
    names = {entry.canonical for entry in updated.terms}
    assert {"Aang", "Katara"} <= names
    assert service.processor is not None
    assert (tmp_path / "profiles" / "avatar.yaml").is_file()


def test_shipped_repository_returns_none_when_missing(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli_module.ProfileRepository,
        "default",
        classmethod(lambda cls: (_ for _ in ()).throw(FileNotFoundError("missing"))),
    )
    assert cli_module._shipped_repository(None) is None
    assert cli_module._shipped_repository(tmp_path) is not None


def test_visual_service_uses_fps_for_sampling_timing_and_cache(tmp_path: Path) -> None:
    one_fps = cli_module._visual_service(None, None, tmp_path / "one", 1.0)
    two_fps = cli_module._visual_service(None, None, tmp_path / "two", 2.0)
    short_cards = cli_module._visual_service(None, None, tmp_path / "short", 1.0, 2)

    assert one_fps._visual_pipeline._sampler._interval == 1.0
    assert one_fps._visual_pipeline._tracker._frame_interval == 1.0
    assert two_fps._visual_pipeline._sampler._interval == 0.5
    assert two_fps._visual_pipeline._tracker._frame_interval == 0.5
    assert one_fps._visual_key != two_fps._visual_key
    assert one_fps._visual_key != short_cards._visual_key
    one_fps.close()
    two_fps.close()
    short_cards.close()
