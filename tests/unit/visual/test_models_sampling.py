import logging
from pathlib import Path

import numpy as np
import pytest

from subtitlegen.visual.models import (
    BoundingBox,
    OcrResult,
    SampledFrame,
    StyledCue,
    VisualEvent,
    VisualObservation,
)
from subtitlegen.visual.pipeline import VisualTextPipeline
from subtitlegen.visual.presence import PresenceDecision
from subtitlegen.visual.sampler import AdaptiveVisualSampler, FrameSampler
from subtitlegen.visual.settings import VisualPipelineSettings
from subtitlegen.visual.tracker import perceptual_hash


def test_visual_domain_models_validate_and_box_iou() -> None:
    box = BoundingBox(0, 0, 10, 10, 0.8)
    assert box.area == 100
    assert BoundingBox(0, 0, 6, 18).is_vertical()
    assert not BoundingBox(0, 0, 20, 6).is_vertical()
    assert box.intersection_over_union(BoundingBox(5, 0, 10, 10)) == pytest.approx(1 / 3)
    SampledFrame(0, object())
    OcrResult("日本", 0.9)
    observation = VisualObservation(0, box, "日本", "Japan", 1)
    assert observation.translated_text == "Japan"
    VisualEvent(0, 1, "日本", "Japan", box)
    StyledCue(0, 1, "Japan", "OnScreen")

    invalid = [
        lambda: BoundingBox(-1, 0, 1, 1),
        lambda: BoundingBox(0, 0, 0, 1),
        lambda: BoundingBox(0, 0, 1, 1, 2),
        lambda: SampledFrame(-1, object()),
        lambda: OcrResult("text", 2),
        lambda: VisualObservation(0, box, "", "Japan", 1),
        lambda: VisualEvent(1, 1, "日本", "Japan", box),
        lambda: StyledCue(0, 1, "", "OnScreen"),
    ]
    for create in invalid:
        with pytest.raises(ValueError):
            create()


def test_frame_sampler_combines_regular_and_scene_change_frames(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.touch()
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    white = np.full((32, 32, 3), 255, dtype=np.uint8)
    frames = [(0.0, black), (0.2, black), (0.7, white), (1.1, white)]
    sampler = FrameSampler(
        frames_per_second=1,
        scene_threshold=0.2,
        frame_reader=lambda _path: frames,
    )

    sampled = tuple(sampler.sample(media))

    assert [frame.timestamp for frame in sampled] == [0.0, 0.7, 1.1]
    assert sampled[1].scene_change
    assert sampled[0].interval
    assert not sampled[1].interval
    assert sampled[2].interval
    with pytest.raises(FileNotFoundError):
        tuple(sampler.sample(tmp_path / "missing.mp4"))
    with pytest.raises(ValueError):
        FrameSampler(frames_per_second=3)
    with pytest.raises(ValueError):
        FrameSampler(scene_threshold=0)


def test_adaptive_sampler_probes_then_densifies_around_japanese_hits(
    tmp_path: Path,
) -> None:
    media = tmp_path / "video.mp4"
    media.touch()
    blank = np.zeros((32, 32, 3), dtype=np.uint8)
    titled = np.full((32, 32, 3), 255, dtype=np.uint8)
    frames = [
        (0.0, blank),
        (8.0, titled),
        (8.5, titled),
        (9.0, titled),
        (20.0, blank),
    ]

    class HitScanner:
        def contains_japanese(self, image: object) -> bool:
            return bool(np.asarray(image).any())

    sampled = tuple(
        AdaptiveVisualSampler(
            HitScanner(),
            frames_per_second=2,
            probe_interval_seconds=8,
            refine_window_seconds=1,
            scene_threshold=0.9,
            frame_reader=lambda _path: frames,
        ).sample(media)
    )

    assert [frame.timestamp for frame in sampled] == [8.0, 8.5, 9.0]
    assert AdaptiveVisualSampler(HitScanner())._probe._skip_nonref_frames is False
    empty = tuple(
        AdaptiveVisualSampler(
            HitScanner(),
            frames_per_second=1,
            probe_interval_seconds=8,
            refine_window_seconds=1,
            scene_threshold=0.9,
            frame_reader=lambda _path: [(0.0, blank), (20.0, blank)],
        ).sample(media)
    )
    assert empty == ()
    with pytest.raises(ValueError):
        AdaptiveVisualSampler(HitScanner(), probe_interval_seconds=0)


def test_adaptive_sampler_attaches_title_boxes_to_refine_frames(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.touch()
    blank = np.zeros((32, 32, 3), dtype=np.uint8)
    titled = np.full((32, 32, 3), 255, dtype=np.uint8)
    box = BoundingBox(2, 4, 8, 6)

    class BoxScanner:
        def inspect(self, image: object) -> PresenceDecision:
            if np.asarray(image).any():
                return PresenceDecision(
                    True,
                    "hit",
                    1,
                    ("ドレスローザ",),
                    boxes=(box,),
                )
            return PresenceDecision(False, "no_japanese", 0, ())

    sampled = tuple(
        AdaptiveVisualSampler(
            BoxScanner(),
            frames_per_second=2,
            probe_interval_seconds=8,
            refine_window_seconds=1,
            scene_threshold=0.9,
            frame_reader=lambda _path: [
                (0.0, blank),
                (8.0, titled),
                (8.5, titled),
                (9.0, titled),
                (20.0, blank),
            ],
        ).sample(media)
    )

    assert [frame.timestamp for frame in sampled] == [8.0, 8.5, 9.0]
    assert all(frame.hint_boxes == (box,) for frame in sampled)


def test_frame_sampler_respects_allowed_windows(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.touch()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    sampled = tuple(
        FrameSampler(
            frames_per_second=1,
            scene_threshold=1,
            frame_reader=lambda _path: [(0.0, image), (1.0, image), (2.0, image)],
            allowed_windows=((0.9, 1.1),),
        ).sample(media)
    )
    assert [frame.timestamp for frame in sampled] == [1.0]


def test_perceptual_hash_is_deterministic_and_rejects_empty_images() -> None:
    image = np.arange(81, dtype=np.uint8).reshape(9, 9)
    assert perceptual_hash(image) == perceptual_hash(image.copy())
    with pytest.raises(ValueError):
        perceptual_hash(np.array([]))


def test_visual_pipeline_settings_validate_runtime_overrides() -> None:
    settings = VisualPipelineSettings(frames_per_second=2, minimum_japanese_characters=2)
    assert settings.frame_interval_seconds == 0.5
    assert settings.probe_interval_seconds == 3.0
    assert settings.refine_window_seconds == 12.0
    assert settings.refine_interval_seconds == 1.0
    assert settings.proposal_padding_ratio == pytest.approx(0.08)
    assert settings.skip_nonref_frames is False
    assert settings.cache_identity()
    with pytest.raises(ValueError):
        VisualPipelineSettings(frames_per_second=0.5)
    with pytest.raises(ValueError):
        VisualPipelineSettings(probe_interval_seconds=0)
    with pytest.raises(ValueError):
        VisualPipelineSettings(minimum_japanese_characters=0)
    with pytest.raises(ValueError):
        VisualPipelineSettings(refine_interval_seconds=0)


def test_adaptive_sampler_keeps_disjoint_title_windows_apart() -> None:
    class HitScanner:
        def contains_japanese(self, _image: object) -> bool:
            return True

    sampler = AdaptiveVisualSampler(HitScanner(), refine_window_seconds=12)
    newspaper = BoundingBox(40, 20, 80, 30)
    board = BoundingBox(10, 200, 90, 20)
    split = sampler._windows(
        (
            (100.0, (newspaper,)),
            (110.0, (board,)),
        )
    )
    assert len(split) == 2
    merged = sampler._windows(
        (
            (100.0, (newspaper,)),
            (110.0, (newspaper,)),
        )
    )
    assert len(merged) == 1


def test_adaptive_sampler_skips_scene_change_probe_ocr(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.touch()
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    white = np.full((32, 32, 3), 255, dtype=np.uint8)
    inspected: list[int] = []

    class RecordingScanner:
        def inspect(self, image: object) -> PresenceDecision:
            inspected.append(int(np.asarray(image).mean()))
            return PresenceDecision(False, "no_japanese", 0, ())

    tuple(
        AdaptiveVisualSampler(
            RecordingScanner(),
            frames_per_second=1,
            probe_interval_seconds=4,
            refine_window_seconds=1,
            scene_threshold=0.2,
            frame_reader=lambda _path: [
                (0.0, black),
                (0.3, white),
                (4.0, black),
            ],
        ).sample(media)
    )

    assert inspected == [0, 0]


def test_adaptive_sampler_probe_logs_are_debug(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    media = tmp_path / "video.mp4"
    media.touch()
    blank = np.zeros((8, 8, 3), dtype=np.uint8)

    class MissScanner:
        def contains_japanese(self, _image: object) -> bool:
            return False

    sampler = AdaptiveVisualSampler(
        MissScanner(),
        probe_interval_seconds=8,
        refine_window_seconds=1,
        scene_threshold=0.9,
        frame_reader=lambda _path: [(0.0, blank), (20.0, blank)],
    )
    with caplog.at_level(logging.INFO, logger="subtitlegen.visual.sampler"):
        tuple(sampler.sample(media))
    assert "title-probe" not in caplog.text
    with caplog.at_level(logging.DEBUG, logger="subtitlegen.visual.sampler"):
        tuple(sampler.sample(media))
    assert "title-probe" in caplog.text


def test_adaptive_sampler_prefetch_reuses_probe_decode(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.touch()
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    titled = np.full((8, 8, 3), 255, dtype=np.uint8)
    reads = {"count": 0}

    class HitScanner:
        def contains_japanese(self, image: object) -> bool:
            return bool(np.asarray(image).any())

    def reader(_path: Path) -> list[tuple[float, object]]:
        reads["count"] += 1
        return [(0.0, blank), (8.0, titled), (8.5, titled), (9.0, titled), (20.0, blank)]

    sampler = AdaptiveVisualSampler(
        HitScanner(),
        frames_per_second=2,
        probe_interval_seconds=8,
        refine_window_seconds=1,
        scene_threshold=0.9,
        frame_reader=reader,
    )
    sampler.prefetch_probe(media)
    pending = sampler._prefetches[media.resolve()]
    assert pending._thread is not None
    pending._thread.join(timeout=2)
    after_prefetch = reads["count"]
    sampled = tuple(sampler.sample(media))
    assert [frame.timestamp for frame in sampled] == [8.0, 8.5, 9.0]
    assert after_prefetch == 1
    assert reads["count"] == 2
    sampler.close()


def test_fingerprint_uses_thumbnail_not_full_pixels() -> None:
    import inspect

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    same = VisualTextPipeline._fingerprint(image)
    assert same == VisualTextPipeline._fingerprint(image.copy())
    source = inspect.getsource(VisualTextPipeline._fingerprint)
    assert "perceptual_hash" in source
    assert "tobytes" not in source


def _write_solid_video(path: Path, *, frames: int, fps: int = 10, size: int = 16) -> None:
    import av

    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = size
    stream.height = size
    stream.pix_fmt = "yuv420p"
    for index in range(frames):
        rgb = np.full((size, size, 3), (index * 17) % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_frame_sampler_seeks_refine_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import av

    video = tmp_path / "clip.mp4"
    _write_solid_video(video, frames=50, fps=10)
    seeks: list[int] = []
    real_open = av.open

    class SeekRecorder:
        def __init__(self, container: object) -> None:
            self._container = container

        def seek(self, *seek_args: object, **seek_kwargs: object) -> object:
            if seek_args:
                seeks.append(int(seek_args[0]))  # type: ignore[arg-type]
            return self._container.seek(*seek_args, **seek_kwargs)

        def __enter__(self) -> object:
            self._container.__enter__()
            return self

        def __exit__(self, *exc: object) -> object:
            return self._container.__exit__(*exc)

        def __getattr__(self, name: str) -> object:
            return getattr(self._container, name)

    def wrapped(path: str, *args: object, **kwargs: object) -> SeekRecorder:
        return SeekRecorder(real_open(path, *args, **kwargs))

    monkeypatch.setattr(av, "open", wrapped)
    sampled = tuple(
        FrameSampler(
            interval_seconds=1.0,
            scene_threshold=1.0,
            allowed_windows=((3.0, 4.0),),
        ).sample(video)
    )
    assert seeks
    assert sampled
    assert all(3.0 <= frame.timestamp <= 4.0 + 1e-3 for frame in sampled)
