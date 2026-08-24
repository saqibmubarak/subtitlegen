from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Protocol, cast

import numpy as np

from subtitlegen.media import format_timecode
from subtitlegen.runtime.executor import DECODE_PREFETCH
from subtitlegen.visual.models import BoundingBox, SampledFrame
from subtitlegen.visual.presence import PresenceDecision

logger = logging.getLogger(__name__)

PROBE_PREFETCH_LIMIT = 2


class FrameReader(Protocol):
    def __call__(self, media_path: Path) -> Iterable[tuple[float, Any]]:
        """Yield timestamped RGB arrays in presentation order."""


class FrameSource(Protocol):
    def sample(self, media_path: Path) -> Iterable[SampledFrame]:
        """Yield selected frames for visual processing."""


class JapanesePresenceScanner(Protocol):
    def contains_japanese(self, image: Any) -> bool:
        """Return whether a frame contains any Japanese characters."""


class FrameSampler:
    def __init__(
        self,
        *,
        frames_per_second: float | None = 1.5,
        interval_seconds: float | None = None,
        scene_threshold: float = 0.28,
        frame_reader: FrameReader | None = None,
        allowed_windows: tuple[tuple[float, float], ...] | None = None,
        skip_nonref_frames: bool = False,
        emit_scene_images: bool = True,
    ) -> None:
        if interval_seconds is not None:
            if interval_seconds <= 0:
                raise ValueError("frame interval must be positive")
            self._interval = interval_seconds
        else:
            if frames_per_second is None or not 1 <= frames_per_second <= 2:
                raise ValueError("visual sampling rate must be between one and two fps")
            self._interval = 1 / frames_per_second
        if not 0 < scene_threshold <= 1:
            raise ValueError("scene threshold must be in (0, 1]")
        self._scene_threshold = scene_threshold
        self._frame_reader = frame_reader
        self._allowed_windows = allowed_windows
        self._skip_nonref_frames = skip_nonref_frames
        self._emit_scene_images = emit_scene_images

    def sample(self, media_path: Path) -> Iterable[SampledFrame]:
        if not media_path.is_file():
            raise FileNotFoundError(media_path)
        if self._frame_reader is None:
            yield from self._sample_video(media_path)
            return
        yield from self._sample_arrays(self._frame_reader(media_path))

    def _in_window(self, timestamp: float) -> bool:
        if self._allowed_windows is None:
            return True
        return any(start <= timestamp <= end for start, end in self._allowed_windows)

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
            if not self._in_window(timestamp):
                continue
            if regular:
                yield SampledFrame(
                    max(0.0, timestamp),
                    image,
                    scene_change,
                    interval=True,
                )
            elif scene_change and self._emit_scene_images:
                yield SampledFrame(
                    max(0.0, timestamp),
                    image,
                    True,
                    interval=False,
                )

    def _sample_video(self, media_path: Path) -> Iterable[SampledFrame]:
        if self._allowed_windows:
            yield from self._sample_video_windows(media_path)
            return
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
            if self._skip_nonref_frames:
                stream.codec_context.skip_frame = "NONREF"
            for frame in container.decode(stream):
                if frame.pts is None or stream.time_base is None:
                    continue
                timestamp = float(frame.pts * stream.time_base)
                if not self._in_window(timestamp):
                    continue
                regular = timestamp + 1e-6 >= next_regular
                scan_scene = timestamp + 1e-6 >= next_scene_scan
                if not regular and not scan_scene:
                    continue
                scene_change = False
                if scan_scene:
                    next_scene_scan = timestamp + 0.25
                    signature = self._frame_signature(frame)
                    scene_change = (
                        previous_signature is not None
                        and float(np.mean(np.abs(signature - previous_signature)))
                        >= self._scene_threshold
                    )
                    previous_signature = signature
                if regular:
                    next_regular = timestamp + self._interval
                    yield SampledFrame(
                        max(0.0, timestamp),
                        frame.to_ndarray(format="rgb24"),
                        scene_change,
                        interval=True,
                    )
                elif scene_change and self._emit_scene_images:
                    yield SampledFrame(
                        max(0.0, timestamp),
                        frame.to_ndarray(format="rgb24"),
                        True,
                        interval=False,
                    )

    def _sample_video_windows(self, media_path: Path) -> Iterable[SampledFrame]:
        try:
            import av
        except ImportError as error:
            raise RuntimeError("PyAV is required for frame sampling") from error
        windows = self._allowed_windows or ()
        with av.open(str(media_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            if self._skip_nonref_frames:
                stream.codec_context.skip_frame = "NONREF"
            if stream.time_base is None:
                return
            for start, end in windows:
                yield from self._decode_seek_window(container, stream, start, end)

    def _decode_seek_window(
        self,
        container: Any,
        stream: Any,
        start: float,
        end: float,
    ) -> Iterable[SampledFrame]:
        offset = int(max(0.0, start) / stream.time_base)
        container.seek(offset, stream=stream, backward=True)
        next_regular = start
        for frame in container.decode(stream):
            if frame.pts is None or stream.time_base is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            if timestamp + 1e-6 < start:
                continue
            if timestamp > end + 1e-6:
                break
            if timestamp + 1e-6 < next_regular:
                continue
            next_regular = timestamp + self._interval
            yield SampledFrame(
                max(0.0, timestamp),
                frame.to_ndarray(format="rgb24"),
                False,
                interval=True,
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


class _ProbePrefetch:
    def __init__(self, media_path: Path) -> None:
        self.path = media_path
        self._queue: Queue[SampledFrame | None] = Queue(maxsize=DECODE_PREFETCH)
        self._error: BaseException | None = None
        self._cancelled = Event()
        self._thread: Thread | None = None

    def start(self, decode: Any) -> None:
        self._thread = Thread(
            target=self._run,
            args=(decode,),
            name="title-probe-prefetch",
            daemon=True,
        )
        self._thread.start()

    def _run(self, decode: Any) -> None:
        try:
            for frame in decode(self.path):
                if self._cancelled.is_set():
                    break
                self._queue.put(frame)
        except BaseException as error:
            self._error = error
        finally:
            self._queue.put(None)

    def frames(self) -> Iterator[SampledFrame]:
        while True:
            item = self._queue.get()
            if item is None:
                if self._error is not None:
                    raise self._error
                return
            yield item

    def cancel(self) -> None:
        self._cancelled.set()
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                break
            if item is None:
                break
        if self._thread is not None:
            self._thread.join(timeout=1)


class AdaptiveVisualSampler:
    """Probe for Japanese characters, then densely sample only around hits."""

    def __init__(
        self,
        scanner: JapanesePresenceScanner,
        *,
        frames_per_second: float = 1.5,
        probe_interval_seconds: float = 3.0,
        refine_window_seconds: float = 12.0,
        refine_interval_seconds: float | None = None,
        scene_threshold: float = 0.28,
        skip_nonref_frames: bool = False,
        frame_reader: FrameReader | None = None,
        allowed_windows: tuple[tuple[float, float], ...] | None = None,
    ) -> None:
        if probe_interval_seconds <= 0 or refine_window_seconds <= 0:
            raise ValueError("probe and refine windows must be positive")
        if refine_interval_seconds is not None and refine_interval_seconds <= 0:
            raise ValueError("refine interval must be positive")
        self._scanner = scanner
        self._frames_per_second = frames_per_second
        self._probe_interval = probe_interval_seconds
        self._refine_window = refine_window_seconds
        self._refine_interval = refine_interval_seconds
        self._scene_threshold = scene_threshold
        self._frame_reader = frame_reader
        self._interval = 1 / frames_per_second
        self._skip_nonref_frames = skip_nonref_frames
        self._prefetches: dict[Path, _ProbePrefetch] = {}
        self.probe_count = 0
        self.hit_count = 0
        self.window_count = 0
        self.refine_count = 0
        self._probe = FrameSampler(
            interval_seconds=probe_interval_seconds,
            scene_threshold=scene_threshold,
            frame_reader=frame_reader,
            skip_nonref_frames=skip_nonref_frames,
            emit_scene_images=False,
            allowed_windows=allowed_windows,
        )

    def sample(self, media_path: Path) -> Iterable[SampledFrame]:
        hits: list[tuple[float, tuple[BoundingBox, ...]]] = []
        for frame in self.iter_probe_frames(media_path):
            decision = self.inspect_frame(frame)
            if decision.accepted:
                hits.append((frame.timestamp, decision.boxes))
        windows = self.windows_from_hits(hits)
        if not windows:
            return
        yield from self.iter_refine_frames(media_path, windows)

    def iter_probe_frames(self, media_path: Path) -> Iterable[SampledFrame]:
        self.probe_count = 0
        for frame in self._consume_probe(media_path):
            if not frame.interval:
                logger.debug(
                    "title-probe %s t=%.3f kind=scene-change decision=signature_only",
                    format_timecode(frame.timestamp),
                    frame.timestamp,
                )
                continue
            self.probe_count += 1
            yield frame

    def inspect_frame(self, frame: SampledFrame) -> PresenceDecision:
        decision = self._inspect(frame.image)
        logger.debug(
            "title-probe %s t=%.3f kind=%s decision=%s boxes=%d skipped_crops=%d "
            "orientation=%s rec=%s",
            format_timecode(frame.timestamp),
            frame.timestamp,
            "scene-change" if frame.scene_change else "interval",
            decision.reason,
            decision.box_count,
            decision.skipped_crops,
            list(decision.orientations),
            list(decision.recognized),
        )
        return decision

    def windows_from_hits(
        self,
        hits: Sequence[tuple[float, tuple[BoundingBox, ...]]],
    ) -> tuple[tuple[float, float, tuple[BoundingBox, ...]], ...]:
        windows = self._windows(hits)
        self.hit_count = len(hits)
        self.window_count = len(windows)
        if not windows:
            logger.debug(
                "title-windows none after %d probe(s)",
                self.probe_count,
            )
            return windows
        logger.debug(
            "title-windows %d from %d probe(s) / %d hit(s): %s",
            len(windows),
            self.probe_count,
            len(hits),
            ", ".join(
                f"{format_timecode(start)}-{format_timecode(end)} crops={len(boxes)}"
                for start, end, boxes in windows
            ),
        )
        return windows

    def iter_refine_frames(
        self,
        media_path: Path,
        windows: tuple[tuple[float, float, tuple[BoundingBox, ...]], ...],
    ) -> Iterable[SampledFrame]:
        self.refine_count = 0
        dense_kwargs: dict[str, Any] = {
            "scene_threshold": 1.0 if self._refine_interval is not None else self._scene_threshold,
            "frame_reader": self._frame_reader,
            "allowed_windows": tuple((start, end) for start, end, _boxes in windows),
            "skip_nonref_frames": self._skip_nonref_frames,
        }
        if self._refine_interval is not None:
            dense = FrameSampler(interval_seconds=self._refine_interval, **dense_kwargs)
        else:
            dense = FrameSampler(
                frames_per_second=self._frames_per_second,
                **dense_kwargs,
            )
        for frame in dense.sample(media_path):
            self.refine_count += 1
            yield SampledFrame(
                frame.timestamp,
                frame.image,
                frame.scene_change,
                self._hints(windows, frame.timestamp),
                interval=frame.interval,
            )

    def prefetch_probe(self, media_path: Path) -> None:
        path = media_path.resolve()
        if path in self._prefetches or len(self._prefetches) >= PROBE_PREFETCH_LIMIT:
            return
        if not path.is_file():
            return
        pending = _ProbePrefetch(path)
        pending.start(self._probe.sample)
        self._prefetches[path] = pending

    def _consume_probe(self, media_path: Path) -> Iterable[SampledFrame]:
        path = media_path.resolve()
        pending = self._prefetches.pop(path, None)
        if pending is not None:
            yield from pending.frames()
            return
        yield from self._probe.sample(path)

    def _inspect(self, image: Any) -> PresenceDecision:
        inspect = getattr(self._scanner, "inspect", None)
        if inspect is not None:
            return cast(PresenceDecision, inspect(image))
        accepted = self._scanner.contains_japanese(image)
        return PresenceDecision(
            accepted,
            "hit" if accepted else "no_japanese",
            0,
            (),
        )

    def close(self) -> None:
        for pending in self._prefetches.values():
            pending.cancel()
        self._prefetches.clear()
        close = getattr(self._scanner, "close", None)
        if close is not None:
            close()

    def _windows(
        self,
        hits: Sequence[tuple[float, tuple[BoundingBox, ...]]],
    ) -> tuple[tuple[float, float, tuple[BoundingBox, ...]], ...]:
        if not hits:
            return ()
        expanded = sorted(
            (
                max(0.0, timestamp - self._refine_window),
                timestamp + self._refine_window,
                boxes,
            )
            for timestamp, boxes in hits
        )
        merged: list[list[Any]] = [
            [expanded[0][0], expanded[0][1], list(expanded[0][2])]
        ]
        for start, end, boxes in expanded[1:]:
            if start <= merged[-1][1] and self._spatially_compatible(merged[-1][2], boxes):
                merged[-1][1] = max(merged[-1][1], end)
                merged[-1][2].extend(boxes)
            else:
                merged.append([start, end, list(boxes)])
        return tuple(
            (float(start), float(end), self._dedupe_boxes(tuple(boxes)))
            for start, end, boxes in merged
        )

    @staticmethod
    def _spatially_compatible(
        left: Sequence[BoundingBox],
        right: Sequence[BoundingBox],
    ) -> bool:
        if not left or not right:
            return True
        for first in left:
            for second in right:
                if first.intersection_over_union(second) >= 0.1:
                    return True
                pad = max(40, min(first.width, second.width, first.height, second.height))
                grown = BoundingBox(
                    max(0, first.x - pad),
                    max(0, first.y - pad),
                    first.width + 2 * pad,
                    first.height + 2 * pad,
                )
                if grown.intersection_over_union(second) > 0:
                    return True
        return False

    @staticmethod
    def _hints(
        windows: tuple[tuple[float, float, tuple[BoundingBox, ...]], ...],
        timestamp: float,
    ) -> tuple[BoundingBox, ...]:
        collected: list[BoundingBox] = []
        for start, end, boxes in windows:
            if start <= timestamp <= end:
                collected.extend(boxes)
        return AdaptiveVisualSampler._dedupe_boxes(tuple(collected))

    @staticmethod
    def _dedupe_boxes(boxes: tuple[BoundingBox, ...]) -> tuple[BoundingBox, ...]:
        selected: list[BoundingBox] = []
        for box in sorted(boxes, key=lambda item: item.area, reverse=True):
            if all(box.intersection_over_union(other) < 0.8 for other in selected):
                selected.append(box)
        return tuple(selected)
