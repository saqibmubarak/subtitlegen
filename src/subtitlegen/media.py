from __future__ import annotations

import shutil
import subprocess
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


def format_timecode(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def extract_video_frame(path: Path, timestamp: float, output_path: Path) -> Path:
    """Seek to a timestamp and write one RGB JPEG/PNG for visual QA."""
    import av
    from PIL import Image

    if timestamp < 0:
        raise ValueError("frame timestamp must be non-negative")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"media has no video stream: {path}")
        stream = container.streams.video[0]
        container.seek(round(timestamp * 1_000_000), backward=True)
        for frame in container.decode(stream):
            if frame.pts is None or stream.time_base is None:
                continue
            if float(frame.pts * stream.time_base) + 1e-3 < timestamp:
                continue
            Image.fromarray(frame.to_ndarray(format="rgb24")).save(output_path)
            return output_path
    raise RuntimeError(f"no video frame near {timestamp:.3f}s in {path}")


def media_duration(path: Path) -> float:
    import av

    with av.open(str(path)) as container:
        if container.duration is None:
            return 0.0
        return float(container.duration / av.time_base)


def load_audio_mono(path: Path, sample_rate: int = 16_000) -> npt.NDArray[np.float32]:
    """Decode audio to mono float32. Prefer ffmpeg; fall back to PyAV."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        try:
            return _load_audio_ffmpeg(path, sample_rate, ffmpeg)
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    return _load_audio_pyav(path, sample_rate)


def _load_audio_ffmpeg(
    path: Path,
    sample_rate: int,
    ffmpeg: str,
) -> npt.NDArray[np.float32]:
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    if not completed.stdout:
        raise ValueError(f"media contains no decodable audio: {path}")
    return np.frombuffer(completed.stdout, dtype=np.float32).copy()


def _load_audio_pyav(path: Path, sample_rate: int) -> npt.NDArray[np.float32]:
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
