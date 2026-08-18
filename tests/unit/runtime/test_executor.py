from pathlib import Path

import pytest

from subtitlegen.runtime.executor import GpuResourceToken, StageExecutor
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
