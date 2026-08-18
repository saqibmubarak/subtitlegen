from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from subtitlegen.visual.models import SampledFrame


class FrameReader(Protocol):
    def __call__(self, media_path: Path) -> Iterable[tuple[float, Any]]:
        """Yield timestamped RGB arrays in presentation order."""


class FrameSource(Protocol):
    def sample(self, media_path: Path) -> Iterable[SampledFrame]:
        """Yield selected frames for visual processing."""


class FrameSampler:
    def __init__(
        self,
        *,
        frames_per_second: float = 1.5,
        scene_threshold: float = 0.28,
        frame_reader: FrameReader | None = None,
    ) -> None:
        if not 1 <= frames_per_second <= 2:
            raise ValueError("visual sampling rate must be between one and two fps")
        if not 0 < scene_threshold <= 1:
            raise ValueError("scene threshold must be in (0, 1]")
        self._interval = 1 / frames_per_second
        self._scene_threshold = scene_threshold
        self._frame_reader = frame_reader

    def sample(self, media_path: Path) -> Iterable[SampledFrame]:
        if not media_path.is_file():
            raise FileNotFoundError(media_path)
        if self._frame_reader is None:
            yield from self._sample_video(media_path)
            return
        yield from self._sample_arrays(self._frame_reader(media_path))

    def _sample_arrays(
        self,
        frames: Iterable[tuple[float, Any]],
    ) -> Iterable[SampledFrame]:
        next_regular = 0.0
        previous_signature: np.ndarray[Any, Any] | None = None
        for timestamp, image in frames:
            signature = self._signature(image)
            scene_change = (
                previous_signature is not None
                and float(np.mean(np.abs(signature - previous_signature)))
                >= self._scene_threshold
            )
            previous_signature = signature
            regular = timestamp + 1e-6 >= next_regular
            if regular:
                next_regular = timestamp + self._interval
            if regular or scene_change:
                yield SampledFrame(max(0.0, timestamp), image, scene_change)

    def _sample_video(self, media_path: Path) -> Iterable[SampledFrame]:
        try:
            import av
        except ImportError as error:
            raise RuntimeError("PyAV is required for frame sampling") from error
        next_regular = 0.0
        next_scene_scan = 0.0
        previous_signature: np.ndarray[Any, Any] | None = None
        with av.open(str(media_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.codec_context.skip_frame = "NONREF"
            for frame in container.decode(stream):
                if frame.pts is None or stream.time_base is None:
                    continue
                timestamp = float(frame.pts * stream.time_base)
                regular = timestamp + 1e-6 >= next_regular
                scan_scene = timestamp + 1e-6 >= next_scene_scan
                if not regular and not scan_scene:
                    continue
                if scan_scene:
                    next_scene_scan = timestamp + 0.25
                    signature = self._frame_signature(frame)
                    scene_change = (
                        previous_signature is not None
                        and float(np.mean(np.abs(signature - previous_signature)))
                        >= self._scene_threshold
                    )
                    previous_signature = signature
                else:
                    scene_change = False
                if regular:
                    next_regular = timestamp + self._interval
                if regular or scene_change:
                    yield SampledFrame(
                        max(0.0, timestamp),
                        frame.to_ndarray(format="rgb24"),
                        scene_change,
                    )

    @staticmethod
    def _frame_signature(frame: Any) -> np.ndarray[Any, Any]:
        width = 144
        height = max(8, round(width * frame.height / frame.width))
        thumbnail = frame.reformat(width=width, height=height, format="gray")
        return np.asarray(thumbnail.to_ndarray(), dtype=np.float32) / 255.0

    @staticmethod
    def _signature(image: Any) -> np.ndarray[Any, Any]:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 3:
            array = array.mean(axis=2)
        return array[::16, ::16] / 255.0

