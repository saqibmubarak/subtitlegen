from pathlib import Path

import pytest

from subtitlegen.domain.models import Transcription
from subtitlegen.errors import BackendUnavailableError
from subtitlegen.runtime.capabilities import DeviceCapabilities
from subtitlegen.runtime.factory import BackendFactory
from subtitlegen.runtime.presets import PresetResolver
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
    no_cuda = BackendFactory(DeviceCapabilities("Linux", "x86_64", 0, False))
    with pytest.raises(BackendUnavailableError):
        no_cuda.create("whisperx", AsrSettings())
    with pytest.raises(BackendUnavailableError):
        no_cuda.create("parakeet", AsrSettings())


def test_presets_resolve_for_apple_cuda_and_cpu() -> None:
    resolver = PresetResolver()
    base = AsrSettings()
    apple = DeviceCapabilities("Darwin", "arm64", 0, True)
    cuda = DeviceCapabilities("Linux", "x86_64", 1, False)
    cpu = DeviceCapabilities("Linux", "x86_64", 0, False)

    assert resolver.resolve("quality", apple, base).settings.model == "large-v3"
    assert resolver.resolve("fast", apple, base).backend == "mlx"
    assert resolver.resolve("quality", cuda, base).backend == "whisperx"
    english = resolver.resolve("english-fast", cuda, AsrSettings(language="ja"))
    assert english.backend == "parakeet"
    assert english.settings.language == "en"
    assert resolver.resolve("fast", cuda, base).settings.compute_type == "float16"
    assert resolver.resolve("quality", cpu, base).backend == "faster-whisper"
    assert resolver.resolve("fast", cpu, base).settings.compute_type == "int8"
    with pytest.raises(ValueError, match="unknown preset"):
        resolver.resolve("missing", cpu, base)
