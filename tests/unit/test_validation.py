from pathlib import Path

import pytest

from subtitlegen.domain.models import Cue
from subtitlegen.validation import TimingReport, analyze_cues, is_valid_srt, parse_srt


def test_timing_report_and_analysis() -> None:
    cues = [Cue(0, 2, "one"), Cue(1.5, 12, "two")]
    report = analyze_cues(cues, duration_limit=8)
    assert report == TimingReport(
        cue_count=2,
        median_duration=6.25,
        max_duration=10.5,
        cues_over_limit=1,
        overlaps=1,
    )


def test_parse_srt(tmp_path: Path) -> None:
    path = tmp_path / "sample.srt"
    path.write_text(
        "1\n00:00:01,000 --> 00:00:02,500\nHello\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nWorld\n",
        encoding="utf-8",
    )
    assert parse_srt(path) == [Cue(1, 2.5, "Hello"), Cue(3, 4, "World")]
    assert is_valid_srt(path)
    empty = tmp_path / "empty.srt"
    empty.touch()
    assert not is_valid_srt(empty)
    assert not is_valid_srt(tmp_path / "missing.srt")
    partial = tmp_path / "partial.srt"
    partial.write_text(path.read_text(encoding="utf-8") + "\n3\nbroken", encoding="utf-8")
    assert not is_valid_srt(partial)


def test_avatar_model_smoke_output_meets_timing_targets() -> None:
    cues = [
        Cue(0.82, 4.70, "As an airbender, it's important that you know when to show restraint."),
        Cue(6.10, 10.12, "Even if you have the power to win, you are the avatar."),
        Cue(10.98, 11.98, "A lot of people..."),
    ]
    report = analyze_cues(cues)
    assert 2 <= report.median_duration <= 4
    assert report.max_duration < 8
    assert report.cues_over_limit == 0
    assert report.overlaps == 0


@pytest.mark.integration
def test_avatar_baseline_documents_timing_failure() -> None:
    sample = Path("samples/legendOfAang/LegendOfAang.srt")
    if not sample.exists():
        pytest.skip("local Avatar sample is unavailable")
    report = analyze_cues(parse_srt(sample))
    assert report.cue_count == 282
    assert report.max_duration > 400
    assert report.cues_over_limit > 100
