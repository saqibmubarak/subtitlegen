import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from subtitlegen.runtime.jobs import (
    SCHEMA_VERSION,
    FileLock,
    JobManifest,
    PortableJobStore,
    StageRecord,
)


def test_stage_and_manifest_values_are_validated() -> None:
    with pytest.raises(ValueError):
        StageRecord("", "pending")
    with pytest.raises(ValueError):
        StageRecord("asr", "complete")
    with pytest.raises(ValueError):
        JobManifest(SCHEMA_VERSION + 1, "job", "video.mp4", "0" * 64)

    manifest = JobManifest(SCHEMA_VERSION, "job", "video.mp4", "0" * 64)
    record = StageRecord("asr", "complete", "words.json")
    updated = manifest.with_stage(record)
    assert manifest.stage("asr") is None
    assert updated.stage("asr") == record


def test_file_lock_guards_cross_process_updates(tmp_path: Path) -> None:
    lock_path = tmp_path / "manifest.lock"
    with FileLock(lock_path):
        assert lock_path.exists()
        with pytest.raises(TimeoutError), FileLock(lock_path, timeout_seconds=0.01):
            pass

    script = (
        "import os, sys\n"
        "from filelock import FileLock\n"
        "lock = FileLock(sys.argv[1])\n"
        "lock.acquire()\n"
        "os._exit(0)\n"
    )
    subprocess.run([sys.executable, "-c", script, str(lock_path)], check=True)
    with FileLock(lock_path, timeout_seconds=0.1):
        assert lock_path.exists()


def test_store_round_trips_and_updates_atomic_manifest(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"media")
    store = PortableJobStore(tmp_path / "jobs")

    manifest = store.create(source)
    same = store.create(source)
    assert same == manifest
    updated = store.update_stage(
        manifest,
        "asr",
        "complete",
        artifact="words.json",
    )
    assert store.load(manifest.job_id) == updated
    assert store.artifact_path(manifest, "words.json").parent == store.job_directory(manifest)
    with pytest.raises(ValueError):
        store.artifact_path(manifest, "../escape")
    with pytest.raises(ValueError):
        store.artifact_path(manifest, "folder\\artifact.json")


def test_store_rejects_missing_source_and_corrupt_manifest(tmp_path: Path) -> None:
    store = PortableJobStore(tmp_path / "jobs")
    with pytest.raises(FileNotFoundError):
        store.create(tmp_path / "missing.mp4")

    source = tmp_path / "video.mp4"
    source.write_bytes(b"media")
    manifest = store.create(source)
    (store.job_directory(manifest) / "manifest.json").write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError):
        store.load(manifest.job_id)


def test_store_preserves_concurrent_stage_updates(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"media")
    store = PortableJobStore(tmp_path / "jobs")
    manifest = store.create(source)

    def update(name: str) -> None:
        store.update_stage(manifest, name, "complete", artifact=f"{name}.json")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(update, ("asr", "subtitle")))

    loaded = store.load(manifest.job_id)
    assert {stage.name for stage in loaded.stages} == {"asr", "subtitle"}


def test_store_migrates_version_one_manifests(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"media")
    store = PortableJobStore(tmp_path / "jobs")
    manifest = store.create(source)
    path = store.job_directory(manifest) / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 1
    data.pop("created_at")
    path.write_text(json.dumps(data), encoding="utf-8")

    migrated = store.load(manifest.job_id)

    assert migrated.schema_version == SCHEMA_VERSION
    assert migrated.created_at
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == SCHEMA_VERSION
    assert persisted["created_at"] == migrated.created_at


def test_store_treats_missing_schema_version_as_version_one(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"media")
    store = PortableJobStore(tmp_path / "jobs")
    manifest = store.create(source)
    path = store.job_directory(manifest) / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("schema_version")
    data.pop("created_at")
    path.write_text(json.dumps(data), encoding="utf-8")

    migrated = store.load(manifest.job_id)

    assert migrated.schema_version == SCHEMA_VERSION
    assert migrated.created_at
