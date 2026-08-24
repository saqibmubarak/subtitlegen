"""Score a titles JSONL against Dressrosa gold or an expected-name list."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--names", type=Path)
    parser.add_argument(
        "--negatives",
        type=Path,
        help="Dressrosa-style YAML; uses reviewed_negative_intervals",
    )
    args = parser.parse_args()
    events = load_events_jsonl(args.events)
    report: dict[str, object] = {"events": len(events)}
    if args.gold is not None:
        report["gold"] = score_annotations(events, load_dressrosa_annotations(args.gold))
    if args.names is not None:
        report["names"] = score_name_overlap(events, load_expected_names(args.names))
    if args.negatives is not None:
        intervals = _negative_intervals(args.negatives)
        report["negatives"] = score_negatives(events, intervals)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _negative_intervals(path: Path) -> list[tuple[float, float]]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    intervals: list[tuple[float, float]] = []
    for sample in data.get("samples", ()):
        for start, end in sample.get("reviewed_negative_intervals", ()):
            intervals.append((float(start), float(end)))
    return intervals


if __name__ == "__main__":
    main()
