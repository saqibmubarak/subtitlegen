from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeviceCapabilities:
    operating_system: str
    architecture: str
    cuda_devices: int
    mlx_available: bool

    @property
    def is_apple_silicon(self) -> bool:
        return self.operating_system == "Darwin" and self.architecture == "arm64"

    @classmethod
    def detect(cls) -> DeviceCapabilities:
        try:
            import ctranslate2

            cuda_devices = ctranslate2.get_cuda_device_count()
        except (ImportError, RuntimeError):
            cuda_devices = 0
        return cls(
            operating_system=platform.system(),
            architecture=platform.machine(),
            cuda_devices=cuda_devices,
            mlx_available=importlib.util.find_spec("mlx_whisper") is not None,
        )
