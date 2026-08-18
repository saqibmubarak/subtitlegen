from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt


def discover_media(input_path: Path, extensions: tuple[str, ...]) -> list[Path]:
    normalized = {extension.lower() for extension in extensions}
    if input_path.is_file():
        return [input_path.resolve()] if input_path.suffix.lower() in normalized else []
    return sorted(
        path.resolve()
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized
    )


def media_duration(path: Path) -> float:
    import av

    with av.open(str(path)) as container:
        if container.duration is None:
            return 0.0
        return float(container.duration / av.time_base)


def load_audio_mono(path: Path, sample_rate: int = 16_000) -> npt.NDArray[np.float32]:
    """Decode audio with PyAV so native runs do not require an ffmpeg executable."""
    import av

    chunks: list[npt.NDArray[np.float32]] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"media has no audio stream: {path}")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        for frame in container.decode(container.streams.audio[0]):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray().reshape(-1).astype(np.float32, copy=False))
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray().reshape(-1).astype(np.float32, copy=False))
    if not chunks:
        raise ValueError(f"media contains no decodable audio: {path}")
    return np.concatenate(chunks)
