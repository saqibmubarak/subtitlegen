from pathlib import Path

import pytest

from subtitlegen.domain.models import Transcription
from subtitlegen.runtime.capabilities import DeviceCapabilities
from subtitlegen.runtime.factory import BackendFactory
from subtitlegen.settings import AsrSettings


class FakeBackend:
    def __init__(self, _settings: AsrSettings) -> None:
        pass

    def transcribe(self, _media_path: Path, *, language: str | None = None) -> Transcription:
        return Transcription((), language or "en")


def test_capabilities_identify_apple_silicon() -> None:
    capabilities = DeviceCapabilities("Darwin", "arm64", 0, True)
    assert capabilities.is_apple_silicon
    assert not DeviceCapabilities("Linux", "x86_64", 1, False).is_apple_silicon


def test_capabilities_detect_returns_current_shape() -> None:
    capabilities = DeviceCapabilities.detect()
    assert capabilities.operating_system
    assert capabilities.architecture
    assert capabilities.cuda_devices >= 0


def test_factory_selects_backend_and_rejects_unsupported_mlx() -> None:
    constructors = {"faster-whisper": FakeBackend, "mlx": FakeBackend}
    apple = BackendFactory(DeviceCapabilities("Darwin", "arm64", 0, True), constructors)
    assert apple.select("auto") == "mlx"
    assert isinstance(apple.create("auto", AsrSettings()), FakeBackend)

    linux = BackendFactory(DeviceCapabilities("Linux", "x86_64", 1, False), constructors)
    assert linux.select("auto") == "faster-whisper"
    with pytest.raises(RuntimeError):
        linux.create("mlx", AsrSettings())
    with pytest.raises(ValueError):
        linux.create("missing", AsrSettings())
