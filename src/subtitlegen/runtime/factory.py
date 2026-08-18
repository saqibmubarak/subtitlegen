from __future__ import annotations

from collections.abc import Callable

from subtitlegen.asr.base import AsrBackend
from subtitlegen.asr.faster_whisper import FasterWhisperBackend
from subtitlegen.asr.mlx_whisper import MlxWhisperBackend
from subtitlegen.asr.parakeet import ParakeetBackend
from subtitlegen.asr.whisperx import WhisperXBackend
from subtitlegen.errors import BackendUnavailableError
from subtitlegen.runtime.capabilities import DeviceCapabilities
from subtitlegen.settings import AsrSettings

BackendConstructor = Callable[[AsrSettings], AsrBackend]


class BackendFactory:
    def __init__(
        self,
        capabilities: DeviceCapabilities,
        constructors: dict[str, BackendConstructor] | None = None,
    ) -> None:
        self._capabilities = capabilities
        self._constructors = constructors or {
            "faster-whisper": FasterWhisperBackend,
            "mlx": MlxWhisperBackend,
            "parakeet": ParakeetBackend,
            "whisperx": WhisperXBackend,
        }

    def create(self, name: str, settings: AsrSettings) -> AsrBackend:
        selected = self.select(name)
        try:
            constructor = self._constructors[selected]
        except KeyError as error:
            available = ", ".join(sorted(self._constructors))
            raise ValueError(f"unknown backend '{selected}'; available: {available}") from error
        return constructor(settings)

    def select(self, name: str) -> str:
        if name != "auto":
            if name == "mlx" and not (
                self._capabilities.is_apple_silicon and self._capabilities.mlx_available
            ):
                raise BackendUnavailableError(
                    "MLX requires Apple Silicon and the mac optional dependencies"
                )
            if name in {"parakeet", "whisperx"} and not self._capabilities.cuda_devices:
                raise BackendUnavailableError(f"{name} requires an NVIDIA CUDA device")
            return name
        if self._capabilities.is_apple_silicon and self._capabilities.mlx_available:
            return "mlx"
        return "faster-whisper"
