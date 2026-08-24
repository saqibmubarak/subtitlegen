from pathlib import Path

from subtitlegen.visual.keep import keep_visual_events, should_keep_event
from subtitlegen.visual.models import BoundingBox, VisualEvent
from subtitlegen.visual.score import (
    dump_events_jsonl,
    load_dressrosa_annotations,
    load_events_jsonl,
    load_expected_names,
    score_annotations,
    score_name_overlap,
)


def _event(
    source: str,
    translation: str,
    *,
    start: float = 1,
    end: float = 2,
    width: int = 200,
    height: int = 40,
) -> VisualEvent:
    return VisualEvent(
        start,
        end,
        source,
        translation,
        BoundingBox(10, 800, width, height),
    )


def test_keep_filter_keeps_gold_cards_and_drops_dialogue_ocr() -> None:
    gold = _event("ドレスローザ", "Dressrosa")
    kinemon = _event("一人はぐれた錦えもん", "Kin'emon Gets Separated")
    factory = _event("工場破壊＆侍救出チーム", "Factory Destruction & Samurai Rescue Team")
    garbage = _event("サンジめし", "I'm Not Going Two be a Big fan.")
    filler = _event("そういえば", "By the way")
    date = _event("２０１８年１０月２０日", "The following is the text of the letter:")
    hud = _event("インターネット", "The Internet.")
    chinese = _event("宾牌营", "The guest house.")
    repeated = _event("ミミミ", "I'm not going to lie.")
    newspaper = VisualEvent(
        698,
        703,
        "キッドメアースカーンス",
        "Kid Mears is a good one.",
        BoundingBox(440, 206, 312, 134),
    )
    kept = keep_visual_events(
        (gold, kinemon, factory, garbage, filler, date, hud, chinese, repeated, newspaper),
        glossary=("Dressrosa", "Kin'emon"),
    )
    assert [event.source_text for event in kept] == [
        "ドレスローザ",
        "一人はぐれた錦えもん",
        "工場破壊＆侍救出チーム",
    ]
    assert should_keep_event(gold)
    assert not should_keep_event(filler)
    assert not should_keep_event(date)
    assert not should_keep_event(hud)
    assert not should_keep_event(chinese)
    assert not should_keep_event(newspaper)


def test_score_dressrosa_fixture_and_wci_names(tmp_path: Path) -> None:
    gold = Path("tests/fixtures/dressrosa_visual_annotations.yaml")
    names = Path("tests/fixtures/whole_cake_island_01_expected_names.yaml")
    annotations = load_dressrosa_annotations(gold)
    events = (
        _event("ドレスローザ", "Dressrosa", start=1241.8, end=1242.6),
        _event("ホールケーキアイランド", "Whole Cake Island", start=10, end=12),
    )
    path = tmp_path / "events.titles.jsonl"
    dump_events_jsonl(path, events)
    loaded = load_events_jsonl(path)
    dressrosa = score_annotations(loaded, [annotations[0]])
    assert dressrosa["hits"] == 1
    overlap = score_name_overlap(loaded, load_expected_names(names))
    assert overlap["hits"] >= 2


def test_sweep_picks_keep_filter_over_raw_event_count(tmp_path: Path) -> None:
    gold_event = _event("ドレスローザ", "Dressrosa", start=1241.8, end=1242.6)
    extra = _event("サンジめし", "I'm Not Going Two be a Big fan.", start=20, end=21)
    baseline = tmp_path / "baseline.titles.jsonl"
    filtered = tmp_path / "filtered.titles.jsonl"
    dump_events_jsonl(baseline, (gold_event, extra))
    dump_events_jsonl(filtered, keep_visual_events((gold_event, extra)))
    annotations = load_dressrosa_annotations(
        Path("tests/fixtures/dressrosa_visual_annotations.yaml")
    )
    noisy = score_annotations(load_events_jsonl(baseline), [annotations[0]])
    clean = score_annotations(load_events_jsonl(filtered), [annotations[0]])
    assert noisy["hits"] == clean["hits"] == 1
    assert clean["events"] < noisy["events"]


def test_keep_filter_collapses_duplicate_map_ocr() -> None:
    first = _event("ドレスローザ", "Dressrosa", start=1241.0, end=1242.0)
    second = _event("トレスローザ", "Dressrosa", start=1242.0, end=1246.0)
    other = _event("グリーンビット", "Green Bit", start=1243.0, end=1246.0)
    kept = keep_visual_events((first, second, other))
    assert [event.source_text for event in kept] == ["ドレスローザ", "グリーンビット"]


def test_score_annotation_hits_when_ocr_adds_subtitle_text() -> None:
    annotations = load_dressrosa_annotations(
        Path("tests/fixtures/dressrosa_visual_annotations.yaml")
    )
    soldier = next(item for item in annotations if item["id"] == "one-legged-soldier")
    event = _event(
        "片足の兵隊 通称：怒りの雷兵",
        "One-legged soldiers, known as the Angry Thunder.",
        start=1053.0,
        end=1058.0,
    )
    assert score_annotations((event,), [soldier])["hits"] == 1
