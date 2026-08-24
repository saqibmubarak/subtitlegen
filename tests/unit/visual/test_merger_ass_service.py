from pathlib import Path

import pytest

from subtitlegen.domain.models import Cue
from subtitlegen.export.ass import AssWriter, is_valid_ass, parse_ass_events
from subtitlegen.export.srt import SrtWriter
from subtitlegen.runtime.executor import StageExecutor
from subtitlegen.runtime.jobs import PortableJobStore
from subtitlegen.visual.merger import SubtitleMerger
from subtitlegen.visual.models import BoundingBox, StyledCue, VisualEvent
from subtitlegen.visual.service import MultimodalSubtitleService

GOLDEN = Path(__file__).parents[2] / "fixtures" / "dialogue_visual.ass"


def _visual() -> VisualEvent:
    return VisualEvent(
        0.5,
        2,
        "ドレスローザ",
        "Dressrosa",
        BoundingBox(0, 0, 10, 10),
    )


def test_merger_preserves_simultaneous_dialogue_and_visual_events() -> None:
    merged = SubtitleMerger().merge([Cue(0, 1.24, "Hello")], [_visual()])
    assert [cue.style for cue in merged] == ["Dialogue", "OnScreen"]
    assert merged[0].start == 0
    assert merged[1].start == 0.5


def test_ass_writer_matches_golden_and_round_trips(tmp_path: Path) -> None:
    cues = [
        StyledCue(0, 1.24, "Hello", "Dialogue"),
        StyledCue(0.5, 2, "Dressrosa", "OnScreen"),
    ]
    writer = AssWriter()
    rendered = writer.render(cues)
    assert rendered == GOLDEN.read_text(encoding="utf-8")
    assert parse_ass_events(rendered) == tuple(cues)

    output = tmp_path / "folder" / "output.ass"
    writer.write(cues, output)
    assert output.read_text(encoding="utf-8") == rendered
    escaped = [StyledCue(0, 1, "literal\\N\n{tag}", "OnScreen")]
    assert parse_ass_events(writer.render(escaped)) == tuple(escaped)
    with pytest.raises(ValueError):
        parse_ass_events("Dialogue: malformed")
    assert is_valid_ass(output)
    empty = tmp_path / "empty.ass"
    empty.touch()
    assert not is_valid_ass(empty)
    assert not is_valid_ass(tmp_path / "missing.ass")
    junk = tmp_path / "junk.ass"
    junk.write_text("not ass", encoding="utf-8")
    assert not is_valid_ass(junk)


class FakeVisualPipeline:
    def __init__(self) -> None:
        self.closed = False
        self.calls = 0

    def process(self, _media: Path) -> tuple[VisualEvent, ...]:
        self.calls += 1
        return (_visual(),)

    def close(self) -> None:
        self.closed = True


def test_multimodal_service_writes_atomic_ass_and_releases_models(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.touch()
    dialogue = tmp_path / "video.srt"
    SrtWriter().write([Cue(0, 1.24, "Hello")], dialogue)
    visual = FakeVisualPipeline()
    service = MultimodalSubtitleService(visual, SubtitleMerger(), AssWriter())
    output = tmp_path / "video.ass"

    result = service.process(media, dialogue, output)

    assert result.dialogue_cues == 1
    assert result.visual_events == 1
    assert parse_ass_events(output.read_text(encoding="utf-8"))[1].style == "OnScreen"
    assert output.with_suffix(".titles.jsonl").is_file()
    assert not list(tmp_path.glob("*.tmp"))
    service.close()
    assert visual.closed


def test_multimodal_service_writes_titles_without_dialogue(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.touch()
    visual = FakeVisualPipeline()
    output = tmp_path / "video.ass"

    result = MultimodalSubtitleService(visual, SubtitleMerger(), AssWriter()).process(
        media,
        None,
        output,
    )

    assert result.dialogue_cues == 0
    assert result.visual_events == 1
    events = parse_ass_events(output.read_text(encoding="utf-8"))
    assert [cue.style for cue in events] == ["OnScreen"]


def test_multimodal_service_resumes_valid_visual_artifact(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.touch()
    dialogue = tmp_path / "video.srt"
    SrtWriter().write([Cue(0, 1.24, "Hello")], dialogue)
    visual = FakeVisualPipeline()
    store = PortableJobStore(tmp_path / "jobs")
    service = MultimodalSubtitleService(
        visual,
        SubtitleMerger(),
        AssWriter(),
        store=store,
        executor=StageExecutor(store),
    )
    service.process(media, dialogue, tmp_path / "first.ass")
    service.process(media, dialogue, tmp_path / "second.ass")
    assert visual.calls == 1

    manifest = store.create(media)
    artifact = store.artifact_path(manifest, manifest.stage("visual-v1").artifact or "")
    artifact.write_text("corrupt", encoding="utf-8")
    service.process(media, dialogue, tmp_path / "third.ass")
    assert visual.calls == 2
