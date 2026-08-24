from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from subtitlegen.evaluation.metrics import character_error_rate
from subtitlegen.visual.models import BoundingBox, VisualEvent


def load_events_jsonl(path: Path) -> tuple[VisualEvent, ...]:
    events: list[VisualEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(event_from_record(json.loads(line)))
    return tuple(events)


def dump_events_jsonl(path: Path, events: Sequence[VisualEvent]) -> None:
    path.write_text(
        "".join(json.dumps(event_record(event), ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )


def event_record(event: VisualEvent) -> dict[str, Any]:
    return {
        "start": event.start,
        "end": event.end,
        "source_text": event.source_text,
        "translated_text": event.translated_text,
        "orientation": "vertical" if event.box.is_vertical() else "horizontal",
        "box": {
            "x": event.box.x,
            "y": event.box.y,
            "width": event.box.width,
            "height": event.box.height,
            "score": event.box.score,
        },
    }


def event_from_record(data: dict[str, Any]) -> VisualEvent:
    box = data.get("box") or {}
    return VisualEvent(
        start=float(data["start"]),
        end=float(data["end"]),
        source_text=str(data["source_text"]),
        translated_text=str(data["translated_text"]),
        box=BoundingBox(
            int(box.get("x", 1)),
            int(box.get("y", 1)),
            int(box.get("width", 1)),
            int(box.get("height", 1)),
            float(box.get("score", 1.0)),
        ),
    )


def load_dressrosa_annotations(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    annotations: list[dict[str, Any]] = []
    for sample in data.get("samples", ()):
        video = sample.get("video")
        for annotation in sample.get("annotations", ()):
            item = dict(annotation)
            item["video"] = video
            annotations.append(item)
    return annotations


def load_expected_names(path: Path) -> tuple[str, ...]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names", data) if isinstance(data, dict) else data
    return tuple(str(name) for name in names)


def score_annotations(
    events: Sequence[VisualEvent],
    annotations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    hits = 0
    details: list[dict[str, Any]] = []
    for annotation in annotations:
        match = _best_annotation_match(events, annotation)
        details.append({"id": annotation.get("id"), "hit": match is not None})
        if match is not None:
            hits += 1
    return {
        "annotations": len(annotations),
        "events": len(events),
        "hits": hits,
        "recall": hits / len(annotations) if annotations else 0.0,
        "details": details,
    }


def score_name_overlap(
    events: Sequence[VisualEvent],
    names: Iterable[str],
) -> dict[str, Any]:
    expected = [name for name in names if name]
    matched = [
        name
        for name in expected
        if any(_name_in_event(name, event) for event in events)
    ]
    return {
        "expected": len(expected),
        "events": len(events),
        "hits": len(matched),
        "overlap": len(matched) / len(expected) if expected else 0.0,
        "matched": matched,
    }


def score_negatives(
    events: Sequence[VisualEvent],
    intervals: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    false_positives = [
        event
        for event in events
        if any(start <= event.start <= end or start <= event.end <= end for start, end in intervals)
    ]
    return {"intervals": len(intervals), "false_positives": len(false_positives)}


def _best_annotation_match(
    events: Sequence[VisualEvent],
    annotation: dict[str, Any],
) -> VisualEvent | None:
    start = float(annotation["start"])
    end = float(annotation["end"])
    expected = str(annotation.get("source_text") or "")
    translation = str(annotation.get("translation") or "")
    best: tuple[float, VisualEvent] | None = None
    for event in events:
        if event.end < start - 1 or event.start > end + 1:
            continue
        cer = min(
            _text_match(event.source_text, expected),
            _text_match(event.translated_text, translation),
        )
        if cer > 0.4:
            continue
        if best is None or cer < best[0]:
            best = (cer, event)
    return None if best is None else best[1]


def _text_match(actual: str, expected: str) -> float:
    if not expected:
        return 1.0
    if expected in actual or actual in expected:
        return 0.0
    return character_error_rate(actual, expected)


def _name_in_event(name: str, event: VisualEvent) -> bool:
    needle = name.casefold()
    return needle in event.source_text.casefold() or needle in event.translated_text.casefold()
