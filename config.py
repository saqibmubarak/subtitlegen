"""Compatibility wrapper for the typed settings loader."""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent / "src"))

from subtitlegen.settings import SettingsLoader


def load_config(config_path: str = "config.ini") -> dict[str, Any]:
    settings = SettingsLoader().load(Path(config_path))
    return {
        "model_identifier": settings.asr.model,
        "device": settings.asr.device,
        "language": settings.asr.language,
        "compute_type": settings.asr.compute_type,
        "video_extensions": list(settings.video_extensions),
        "parallel_workers": settings.parallel_workers,
    }