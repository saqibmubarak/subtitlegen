"""Score title JSONL variants against Dressrosa gold and WCI expected names."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from subtitlegen.visual.score import (
    load_dressrosa_annotations,
    load_events_jsonl,
    load_expected_names,
    score_annotations,
    score_name_overlap,
    score_negatives,
)

GOLD = Path("tests/fixtures/dressrosa_visual_annotations.yaml")
WCI_NAMES = Path("tests/fixtures/whole_cake_island_01_expected_names.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", nargs="+", type=Path)
    args = parser.parse_args()
    annotations = load_dressrosa_annotations(GOLD)
    names = load_expected_names(WCI_NAMES)
    intervals = _negatives(GOLD)
    rows = []
    for path in args.events:
        events = load_events_jsonl(path)
        gold = score_annotations(events, annotations)
        report = {
            "path": str(path),
            "events": len(events),
            "vertical": sum(1 for event in events if event.box.is_vertical()),
            "gold_hits": gold["hits"],
            "gold_recall": gold["recall"],
            "name_overlap": score_name_overlap(events, names)["overlap"],
            "negative_fp": score_negatives(events, intervals)["false_positives"],
        }
        rows.append(report)
    rows.sort(key=lambda row: (row["gold_hits"], row["name_overlap"], -row["negative_fp"]))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def _negatives(path: Path) -> list[tuple[float, float]]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    intervals: list[tuple[float, float]] = []
    for sample in data.get("samples", ()):
        for start, end in sample.get("reviewed_negative_intervals", ()):
            intervals.append((float(start), float(end)))
    return intervals


if __name__ == "__main__":
    main()
