"""Run title-scan variants on local Dressrosa / Whole Cake videos and score them."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from PIL import Image

from subtitlegen.cli import _keep_glossary, _visual_service
from subtitlegen.media import extract_video_frame, format_timecode
from subtitlegen.profiles.repository import ProfileRepository
from subtitlegen.visual.keep import keep_visual_events
from subtitlegen.visual.models import VisualEvent
from subtitlegen.visual.score import (
    dump_events_jsonl,
    event_record,
    load_dressrosa_annotations,
    load_expected_names,
    score_annotations,
    score_name_overlap,
    score_negatives,
)
from subtitlegen.visual.settings import VisualPipelineSettings

GOLD = Path("tests/fixtures/dressrosa_visual_annotations.yaml")
WCI_NAMES = Path("tests/fixtures/whole_cake_island_01_expected_names.yaml")
DRESSROSA_NAMES = Path("tests/fixtures/dressrosa_01_expected_names.yaml")

VARIANTS: dict[str, VisualPipelineSettings] = {
    "baseline": VisualPipelineSettings(
        probe_interval_seconds=4.0,
        minimum_japanese_characters=5,
        probe_analysis_width=960,
        probe_maximum_crops=24,
        tracker_minimum_observations=2,
    ),
    "current": VisualPipelineSettings(),
    "high_recall": VisualPipelineSettings(
        probe_interval_seconds=2.0,
        minimum_japanese_characters=3,
        minimum_vertical_box_area_ratio=0.0004,
        probe_analysis_width=1280,
        probe_maximum_crops=32,
        tracker_minimum_observations=1,
    ),
    "vertical_focus": VisualPipelineSettings(
        probe_interval_seconds=3.0,
        minimum_japanese_characters=4,
        minimum_vertical_box_area_ratio=0.0003,
        probe_analysis_width=1280,
        probe_maximum_crops=32,
    ),
    "precision": VisualPipelineSettings(
        minimum_japanese_characters=6,
        minimum_vertical_box_area_ratio=0.01,
        probe_accept_tall_weak=False,
        probe_analysis_width=480,
        probe_maximum_crops=16,
    ),
}


def gold_windows() -> dict[str, tuple[tuple[float, float], ...]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for annotation in load_dressrosa_annotations(GOLD):
        video = str(annotation["video"])
        start = max(0.0, float(annotation["start"]) - 8)
        end = float(annotation["end"]) + 8
        grouped.setdefault(video, []).append((start, end))
    return {name: tuple(windows) for name, windows in grouped.items()}


def dump_scene_jobs(
    media: Path,
    events: tuple[VisualEvent, ...],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, event in enumerate(events, start=1):
        midpoint = (event.start + event.end) / 2
        frame_path = output_dir / f"{index:03d}_{format_timecode(midpoint).replace(':', '-')}.jpg"
        extract_video_frame(media, midpoint, frame_path)
        crop_path = frame_path.with_name(frame_path.stem + "_crop.jpg")
        image = Image.open(frame_path)
        box = (
            event.box.x,
            event.box.y,
            event.box.x + event.box.width,
            event.box.y + event.box.height,
        )
        image.crop(box).save(crop_path, quality=90)
        record = event_record(event)
        record["frame"] = str(frame_path)
        record["crop"] = str(crop_path)
        frame_path.with_suffix(".json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def score_run(events: tuple[VisualEvent, ...], video_name: str) -> dict[str, object]:
    annotations = [
        item for item in load_dressrosa_annotations(GOLD) if item.get("video") == video_name
    ]
    report: dict[str, object] = {
        "events": len(events),
        "vertical": sum(1 for event in events if event.box.is_vertical()),
    }
    if annotations:
        report["gold"] = score_annotations(events, annotations)
    if "Dressrosa 01" in video_name and DRESSROSA_NAMES.is_file():
        report["names"] = score_name_overlap(events, load_expected_names(DRESSROSA_NAMES))
    if "Whole Cake" in video_name:
        report["names"] = score_name_overlap(events, load_expected_names(WCI_NAMES))
    if "Dressrosa 05" in video_name:
        intervals = []
        for sample in __import__("yaml").safe_load(GOLD.read_text())["samples"]:
            if sample["video"].endswith("05.mp4"):
                intervals = [tuple(item) for item in sample.get("reviewed_negative_intervals", ())]
        report["negatives"] = score_negatives(events, intervals)
    return report


def run_variant(
    *,
    videos: list[Path],
    variant: str,
    settings: VisualPipelineSettings,
    windows: dict[str, tuple[tuple[float, float], ...]] | None,
    root: Path,
    apply_keep_filter: bool,
) -> dict[str, object]:
    profile = None
    shipped = ProfileRepository.default()
    if shipped is not None:
        profile = shipped.load("one-piece")
    cache = root / variant / "cache"
    service = _visual_service(
        profile,
        None,
        cache,
        settings.frames_per_second,
        settings.minimum_japanese_characters,
        settings.probe_interval_seconds,
        settings.refine_window_seconds,
        settings=settings,
        allowed_windows=None,
        apply_keep_filter=apply_keep_filter,
    )
    results: dict[str, object] = {}
    try:
        for video in videos:
            scoped = windows.get(video.name) if windows else None
            if scoped:
                service._visual_pipeline._sampler._probe._allowed_windows = scoped
            else:
                service._visual_pipeline._sampler._probe._allowed_windows = None
            output = root / variant / f"{video.stem}.ass"
            result = service.process(video, video.with_suffix(".srt"), output)
            from subtitlegen.visual.score import load_events_jsonl

            events = load_events_jsonl(output.with_suffix(".titles.jsonl"))
            dump_scene_jobs(video, events, root / variant / f"{video.stem}.scene-jobs")
            filtered = keep_visual_events(events, glossary=_keep_glossary(profile))
            results[video.name] = {
                "raw": score_run(events, video.name),
                "filtered": score_run(filtered, video.name),
                "ass": str(output),
                "visual_events": result.visual_events,
            }
    finally:
        service.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS))
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(".subtitlegen/sweeps"))
    args = parser.parse_args()
    os.environ.setdefault("SUBTITLEGEN_ENRICH_GLOSSARY", "0")
    samples = Path("samples")
    if args.full:
        videos = [
            samples / "[Muhn Pace] Dressrosa 01.mp4",
            samples / "[Muhn Pace] Whole Cake Island 01.mp4",
        ]
        windows = None
    else:
        videos = [
            samples / name
            for name in gold_windows()
            if (samples / name).is_file()
        ]
        windows = gold_windows()
    videos = [video for video in videos if video.is_file()]
    selected = args.variant or ["current", "high_recall"]
    report = {}
    for name in selected:
        print(f"running {name} on {[video.name for video in videos]}", flush=True)
        report[name] = run_variant(
            videos=videos,
            variant=name,
            settings=VARIANTS[name],
            windows=windows,
            root=args.root,
            apply_keep_filter=False,
        )
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
