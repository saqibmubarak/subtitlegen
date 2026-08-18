from pathlib import Path
from typing import Any

import pytest

from subtitlegen.asr.context import AsrContext
from subtitlegen.cues.builder import CueBuilder
from subtitlegen.domain.models import Transcription, Word
from subtitlegen.export.srt import SrtWriter
from subtitlegen.runtime.executor import StageExecutor
from subtitlegen.runtime.jobs import PortableJobStore
from subtitlegen.runtime.service import RuntimeResult, RuntimeService


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.contexts: list[AsrContext | None] = []

    def transcribe(
        self,
        _media_path: Path,
        *,
        language: str | None = None,
        **_kwargs: Any,
    ) -> Transcription:
        self.calls += 1
        self.contexts.append(_kwargs.get("context"))
        return Transcription(
            (Word(0, 1, "Hello"), Word(1, 2, " world.")),
            language or "en",
            2,
        )


def _service(
    tmp_path: Path,
    backend: FakeBackend,
    *,
    output_key: str = "srt-v1",
    context: AsrContext | None = None,
) -> RuntimeService:
    store = PortableJobStore(tmp_path / "jobs")
    return RuntimeService(
        backend,
        CueBuilder(),
        SrtWriter(),
        store,
        StageExecutor(store),
        asr_key="fake-v1",
        output_key=output_key,
        context=context,
    )


def test_runtime_result_is_immutable_value(tmp_path: Path) -> None:
    result = RuntimeResult("skipped", tmp_path / "output.srt", None)
    assert result.status == "skipped"


def test_service_generates_resumes_and_skips(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.write_bytes(b"media")
    output = tmp_path / "output.srt"
    backend = FakeBackend()
    service = _service(tmp_path, backend)

    generated = service.process(media, output, language="en")
    assert generated.status == "generated"
    assert output.exists()

    output.unlink()
    resumed = service.process(media, output, language="en")
    assert resumed.status == "resumed"
    assert backend.calls == 1

    skipped = service.process(media, output, language="en")
    assert skipped.status == "skipped"
    assert skipped.job_id is not None

    changed_rules = _service(tmp_path, backend, output_key="srt-v2")
    assert changed_rules.process(media, output).status == "resumed"
    assert backend.calls == 1


def test_service_injects_asr_context(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.write_bytes(b"media")
    backend = FakeBackend()
    context = AsrContext(prompt="Aang", hotwords=("Aang",))
    _service(tmp_path, backend, context=context).process(media, tmp_path / "output.srt")
    assert backend.contexts == [context]


def test_service_rejects_invalid_run_key(tmp_path: Path) -> None:
    backend = FakeBackend()
    store = PortableJobStore(tmp_path / "jobs")
    with pytest.raises(ValueError):
        RuntimeService(
            backend,
            CueBuilder(),
            SrtWriter(),
            store,
            StageExecutor(store),
            asr_key="../escape",
            output_key="srt-v1",
        )


def test_service_repairs_corrupt_cached_artifacts_and_overwrites(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.write_bytes(b"media")
    output = tmp_path / "output.srt"
    backend = FakeBackend()
    service = _service(tmp_path, backend)
    service.process(media, output)

    words = next((tmp_path / "jobs").glob("**/words.json"))
    words.write_text("{", encoding="utf-8")
    output.unlink()
    service.process(media, output)
    assert backend.calls == 2

    cached_srt = next((tmp_path / "jobs").glob("**/subtitle.srt"))
    cached_srt.write_text("broken", encoding="utf-8")
    output.unlink()
    service.process(media, output)
    assert "Hello world." in output.read_text(encoding="utf-8")
    assert backend.calls == 2

    service.process(media, output, overwrite=True)
    assert backend.calls == 2

    refreshed = service.process(media, output, overwrite=True, refresh_stages=True)
    assert refreshed.status == "generated"
    assert backend.calls == 3
