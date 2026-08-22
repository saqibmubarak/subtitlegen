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
        self.closed = False

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

    def close(self) -> None:
        self.closed = True


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


def test_service_releases_backend(tmp_path: Path) -> None:
    backend = FakeBackend()
    _service(tmp_path, backend).close()
    assert backend.closed


def test_service_prefetches_audio_when_backend_supports_it(tmp_path: Path) -> None:
    class PrefetchBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.prefetched: list[Path] = []

        def prefetch_audio(self, media_path: Path) -> None:
            self.prefetched.append(media_path)

    backend = PrefetchBackend()
    media = tmp_path / "clip.mp4"
    media.touch()
    _service(tmp_path, backend).prefetch_audio(media)
    assert backend.prefetched == [media]


def test_service_can_replace_cue_processor(tmp_path: Path) -> None:
    class Marker:
        def process(self, cues: Any) -> Any:
            return list(cues)

    service = _service(tmp_path, FakeBackend())
    processor = Marker()
    service.set_cue_processor(processor)
    assert service._cue_processor is processor


def test_service_generates_resumes_and_skips(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.write_bytes(b"media")
    output = tmp_path / "output.srt"
    backend = FakeBackend()
    service = _service(tmp_path, backend)

    generated = service.process(media, output, language="en")
    service.flush_writes()
    assert generated.status == "generated"
    assert output.exists()

    output.unlink()
    resumed = service.process(media, output, language="en")
    service.flush_writes()
    assert resumed.status == "resumed"
    assert backend.calls == 1

    skipped = service.process(media, output, language="en")
    assert skipped.status == "skipped"

    changed_rules = _service(tmp_path, backend, output_key="srt-v2")
    assert changed_rules.process(media, output).status == "skipped"
    assert backend.calls == 1
    overwritten = changed_rules.process(media, output, overwrite=True)
    changed_rules.flush_writes()
    assert overwritten.status == "resumed"
    assert backend.calls == 1


def test_service_skips_existing_valid_srt_without_sidecar(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.write_bytes(b"media")
    output = tmp_path / "output.srt"
    output.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    backend = FakeBackend()
    result = _service(tmp_path, backend).process(media, output)
    assert result.status == "skipped"
    assert backend.calls == 0


def test_service_injects_asr_context(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.write_bytes(b"media")
    backend = FakeBackend()
    context = AsrContext(prompt="Aang", hotwords=("Aang",))
    service = _service(tmp_path, backend, context=context)
    service.process(media, tmp_path / "output.srt")
    service.flush_writes()
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
    service.flush_writes()

    words = next((tmp_path / "jobs").glob("**/words.json"))
    words.write_text("{", encoding="utf-8")
    output.unlink()
    service.process(media, output)
    service.flush_writes()
    assert backend.calls == 2

    cached_srt = next((tmp_path / "jobs").glob("**/subtitle.srt"))
    cached_srt.write_text("broken", encoding="utf-8")
    output.unlink()
    service.process(media, output)
    service.flush_writes()
    assert "Hello world." in output.read_text(encoding="utf-8")
    assert backend.calls == 2

    service.process(media, output, overwrite=True)
    service.flush_writes()
    assert backend.calls == 2

    refreshed = service.process(media, output, overwrite=True, refresh_stages=True)
    service.flush_writes()
    assert refreshed.status == "generated"
    assert backend.calls == 3
