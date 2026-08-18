from pathlib import Path

import pytest

from subtitlegen.settings import AppSettings, AsrSettings, SettingsLoader, VadSettings


def test_settings_classes_validate() -> None:
    assert AppSettings().parallel_workers == 1
    with pytest.raises(ValueError):
        AppSettings(parallel_workers=0)
    with pytest.raises(ValueError):
        AsrSettings(model="")
    with pytest.raises(ValueError):
        VadSettings(max_speech_duration_s=0)


def test_loader_reads_legacy_ini_into_typed_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.ini"
    path.write_text(
        """
[MODELS]
quality = large-v3
[TRANSCRIPTION]
model_name = quality
device = cpu
compute_type = int8
language = None
parallel_workers = 1
beam_size = 3
[VAD]
min_silence_duration_ms = 350
[CUES]
max_duration_seconds = 5
[FILES]
video_extensions = .mp4, .webm
""".strip(),
        encoding="utf-8",
    )

    settings = SettingsLoader().load(path)

    assert settings.asr.model == "large-v3"
    assert settings.asr.language is None
    assert settings.asr.beam_size == 3
    assert settings.asr.vad.min_silence_duration_ms == 350
    assert settings.cues.max_duration_seconds == 5
    assert settings.video_extensions == (".mp4", ".webm")


def test_loader_uses_portable_defaults_for_missing_file(tmp_path: Path) -> None:
    settings = SettingsLoader().load(tmp_path / "missing.ini")
    assert settings.asr.model == "large-v3-turbo"
    assert settings.asr.device == "auto"
    assert settings.asr.compute_type == "auto"
