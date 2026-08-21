from __future__ import annotations

from dataclasses import dataclass, replace

from subtitlegen.runtime.capabilities import DeviceCapabilities
from subtitlegen.settings import AsrSettings


@dataclass(frozen=True, slots=True)
class ResolvedPreset:
    name: str
    backend: str
    settings: AsrSettings


class PresetResolver:
    NAMES = ("fast", "quality", "english-fast")

    def resolve(
        self,
        name: str,
        capabilities: DeviceCapabilities,
        base: AsrSettings,
    ) -> ResolvedPreset:
        if name not in self.NAMES:
            available = ", ".join(self.NAMES)
            raise ValueError(f"unknown preset '{name}'; available: {available}")
        if capabilities.is_apple_silicon and capabilities.mlx_available:
            model = "large-v3" if name == "quality" else "large-v3-turbo"
            return ResolvedPreset(name, "mlx", replace(base, model=model, device="auto"))
        if capabilities.cuda_devices:
            if name == "quality":
                if capabilities.whisperx_available:
                    return ResolvedPreset(
                        name,
                        "whisperx",
                        replace(
                            base,
                            model="large-v3",
                            device="cuda",
                            compute_type="float16",
                        ),
                    )
                return ResolvedPreset(
                    name,
                    "faster-whisper",
                    replace(
                        base,
                        model="large-v3",
                        device="cuda",
                        compute_type="float16",
                    ),
                )
            if name == "english-fast":
                return ResolvedPreset(
                    name,
                    "parakeet",
                    replace(
                        base,
                        model="nvidia/parakeet-tdt-0.6b-v3",
                        device="cuda",
                        compute_type="float16",
                        language="en",
                    ),
                )
            return ResolvedPreset(
                name,
                "faster-whisper",
                replace(
                    base,
                    model="large-v3-turbo",
                    device="cuda",
                    compute_type="float16",
                ),
            )
        model = "large-v3" if name == "quality" else "large-v3-turbo"
        return ResolvedPreset(
            name,
            "faster-whisper",
            replace(base, model=model, device="cpu", compute_type="int8"),
        )
