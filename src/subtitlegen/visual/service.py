from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from subtitlegen.export.ass import AssWriter
from subtitlegen.media import extract_video_frame, format_timecode
from subtitlegen.runtime.executor import StageExecutor
from subtitlegen.runtime.jobs import PortableJobStore
from subtitlegen.validation import is_valid_srt, parse_srt
from subtitlegen.visual.merger import SubtitleMerger
from subtitlegen.visual.models import BoundingBox, VisualEvent
from subtitlegen.visual.pipeline import VisualTextPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MultimodalResult:
    output_path: Path
    dialogue_cues: int
    visual_events: int


class MultimodalSubtitleService:
    def __init__(
        self,
        visual_pipeline: VisualTextPipeline,
        merger: SubtitleMerger,
        writer: AssWriter,
        *,
        store: PortableJobStore | None = None,
        executor: StageExecutor | None = None,
        visual_key: str = "visual-v1",
    ) -> None:
        if (store is None) != (executor is None):
            raise ValueError("visual store and executor must be provided together")
        self._visual_pipeline = visual_pipeline
        self._merger = merger
        self._writer = writer
        self._store = store
        self._executor = executor
        self._visual_key = visual_key

    def process(
        self,
        media_path: Path,
        dialogue_srt: Path | None,
        output_path: Path,
    ) -> MultimodalResult:
        dialogue = (
            parse_srt(dialogue_srt)
            if dialogue_srt is not None and is_valid_srt(dialogue_srt)
            else []
        )
        visual = self._load_or_run_visual(media_path)
        self._log_and_preview_titles(media_path, visual, output_path)
        merged = self._merger.merge(dialogue, list(visual))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            self._writer.write(merged, temporary)
            temporary.replace(output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return MultimodalResult(output_path, len(dialogue), len(visual))

    def prefetch_probe(self, media_path: Path) -> None:
        prefetch = getattr(self._visual_pipeline, "prefetch_probe", None)
        if prefetch is not None:
            prefetch(media_path)

    def close(self) -> None:
        self._visual_pipeline.close()

    def _log_and_preview_titles(
        self,
        media_path: Path,
        events: tuple[VisualEvent, ...],
        output_path: Path,
    ) -> None:
        preview_dir = output_path.with_name(f"{output_path.stem}.visual-qa")
        for index, event in enumerate(events, start=1):
            midpoint = (event.start + event.end) / 2
            logger.info(
                "on-screen title %s-%s: %s -> %s",
                format_timecode(event.start),
                format_timecode(event.end),
                event.source_text,
                event.translated_text,
            )
            preview = preview_dir / (
                f"{index:03d}_{format_timecode(midpoint).replace(':', '-')}_"
                f"{_preview_slug(event.translated_text)}.jpg"
            )
            try:
                extract_video_frame(media_path, midpoint, preview)
                logger.info("title screenshot %s", preview)
            except (OSError, RuntimeError, ValueError, ImportError) as error:
                logger.warning("title screenshot failed at %s: %s", preview, error)

    def _load_or_run_visual(self, media_path: Path) -> tuple[VisualEvent, ...]:
        if self._store is None or self._executor is None:
            return self._visual_pipeline.process(media_path)
        manifest = self._store.create(media_path)

        def build(job_directory: Path) -> Path:
            events = self._visual_pipeline.process(media_path)
            output = job_directory / self._visual_key / "events.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".tmp")
            temporary.write_text(
                json.dumps([self._event_data(event) for event in events], sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(output)
            return output

        _, artifact = self._executor.run(
            manifest,
            self._visual_key,
            build,
            validator=self._valid_events,
        )
        data = json.loads(artifact.read_text(encoding="utf-8"))
        return tuple(self._event_from_data(item) for item in data)

    @staticmethod
    def _event_data(event: VisualEvent) -> dict[str, object]:
        return {
            "start": event.start,
            "end": event.end,
            "source_text": event.source_text,
            "translated_text": event.translated_text,
            "category": event.category,
            "box": {
                "x": event.box.x,
                "y": event.box.y,
                "width": event.box.width,
                "height": event.box.height,
                "score": event.box.score,
            },
        }

    @staticmethod
    def _event_from_data(data: dict[str, object]) -> VisualEvent:
        box_data = data["box"]
        if not isinstance(box_data, dict):
            raise ValueError("visual event box must be an object")
        return VisualEvent(
            start=MultimodalSubtitleService._number(data["start"]),
            end=MultimodalSubtitleService._number(data["end"]),
            source_text=MultimodalSubtitleService._text(data["source_text"]),
            translated_text=MultimodalSubtitleService._text(data["translated_text"]),
            category=MultimodalSubtitleService._text(data["category"]),
            box=BoundingBox(
                x=int(MultimodalSubtitleService._number(box_data["x"])),
                y=int(MultimodalSubtitleService._number(box_data["y"])),
                width=int(MultimodalSubtitleService._number(box_data["width"])),
                height=int(MultimodalSubtitleService._number(box_data["height"])),
                score=MultimodalSubtitleService._number(box_data["score"]),
            ),
        )

    @staticmethod
    def _number(value: object) -> float:
        if not isinstance(value, (int, float)):
            raise ValueError("visual event number has an invalid type")
        return float(value)

    @staticmethod
    def _text(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("visual event text has an invalid type")
        return value

    @classmethod
    def _valid_events(cls, path: Path) -> bool:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return isinstance(data, list) and all(
                isinstance(item, dict) and cls._event_from_data(item)
                for item in data
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return False


def _preview_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return (slug or "title")[:40]
