import threading
from pathlib import Path

import pytest

from subtitlegen.runtime.executor import (
    GpuResourceToken,
    PrefetchInferExecutor,
    StageExecutor,
)
from subtitlegen.runtime.jobs import JobManifest, PortableJobStore


def _job(tmp_path: Path) -> tuple[PortableJobStore, JobManifest]:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"media")
    store = PortableJobStore(tmp_path / "jobs")
    return store, store.create(source)


def test_gpu_token_validates_capacity_and_releases() -> None:
    with pytest.raises(ValueError):
        GpuResourceToken(0)
    token = GpuResourceToken()
    with token.acquire():
        assert token is not None


def test_executor_runs_and_resumes_completed_stage(tmp_path: Path) -> None:
    store, manifest = _job(tmp_path)
    executor = StageExecutor(store, GpuResourceToken())
    calls = 0

    def action(job_directory: Path) -> Path:
        nonlocal calls
        calls += 1
        artifact = job_directory / "artifact.json"
        artifact.write_text("{}", encoding="utf-8")
        return artifact

    updated, artifact = executor.run(manifest, "asr", action)
    resumed, resumed_artifact = executor.run(updated, "asr", action)
    assert calls == 1
    assert artifact == resumed_artifact
    assert resumed == updated
    executor.run(
        updated,
        "asr",
        action,
        validator=lambda path: path.read_text(encoding="utf-8") == "{}",
        force=True,
    )
    assert calls == 2


def test_executor_records_failures_and_rejects_external_artifacts(tmp_path: Path) -> None:
    store, manifest = _job(tmp_path)
    executor = StageExecutor(store)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError):
        executor.run(manifest, "asr", lambda _directory: outside)
    loaded = store.load(manifest.job_id)
    stage = loaded.stage("asr")
    assert stage is not None and stage.status == "failed"


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_executor_records_user_cancellation(
    tmp_path: Path,
    interruption: type[BaseException],
) -> None:
    store, manifest = _job(tmp_path)
    executor = StageExecutor(store, GpuResourceToken())

    def cancel(_directory: Path) -> Path:
        raise interruption

    with pytest.raises(interruption):
        executor.run(manifest, "asr", cancel)

    stage = store.load(manifest.job_id).stage("asr")
    assert stage is not None
    assert stage.status == "cancelled"
    assert stage.error == "cancelled by user"

    def recover(job_directory: Path) -> Path:
        artifact = job_directory / "recovered.json"
        artifact.write_text("{}", encoding="utf-8")
        return artifact

    recovered, _ = executor.run(store.load(manifest.job_id), "asr", recover)
    recovered_stage = recovered.stage("asr")
    assert recovered_stage is not None
    assert recovered_stage.status == "complete"


def test_prefetch_infer_runs_one_infer_thread() -> None:
    seen: list[int] = []
    names: set[str] = set()

    def process(batch: list[int]) -> None:
        names.add(threading.current_thread().name)
        seen.extend(batch)

    counted = PrefetchInferExecutor(prefetch=2).run(
        [1, 2, 3, 4, 5],
        process,
        batch_size=2,
    )
    assert counted == 5
    assert seen == [1, 2, 3, 4, 5]
    assert names == {"title-infer"}
    with pytest.raises(ValueError):
        PrefetchInferExecutor(prefetch=0)
    with pytest.raises(RuntimeError, match="boom"):
        PrefetchInferExecutor().run(
            [1, 2],
            lambda _batch: (_ for _ in ()).throw(RuntimeError("boom")),
            batch_size=1,
        )
