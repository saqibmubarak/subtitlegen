from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from filelock import FileLock as PlatformFileLock
from filelock import Timeout

StageStatus = Literal["pending", "running", "complete", "failed"]
SCHEMA_VERSION = 1


class FileLock:
    """OS-backed cross-platform lock released automatically after process death."""

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("lock timeout must be positive")
        self._path = path
        self._lock = PlatformFileLock(path, timeout=timeout_seconds)

    def __enter__(self) -> FileLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lock.acquire()
        except Timeout as error:
            raise TimeoutError(f"timed out waiting for lock: {self._path}") from error
        return self

    def __exit__(self, *_exc: object) -> None:
        self._lock.release()


@dataclass(frozen=True, slots=True)
class StageRecord:
    name: str
    status: StageStatus
    artifact: str | None = None
    error: str | None = None
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must not be empty")
        if self.status == "complete" and not self.artifact:
            raise ValueError("complete stages require an artifact")


@dataclass(frozen=True, slots=True)
class JobManifest:
    schema_version: int
    job_id: str
    source_name: str
    source_sha256: str
    stages: tuple[StageRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema {self.schema_version}")
        if not self.job_id or not self.source_name or len(self.source_sha256) != 64:
            raise ValueError("invalid job identity")

    def stage(self, name: str) -> StageRecord | None:
        return next((stage for stage in self.stages if stage.name == name), None)

    def with_stage(self, record: StageRecord) -> JobManifest:
        stages = tuple(stage for stage in self.stages if stage.name != record.name)
        return replace(self, stages=(*stages, record))


class PortableJobStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def create(self, source: Path) -> JobManifest:
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = self._sha256(source)
        job_id = source_hash[:20]
        manifest_path = self._manifest_path(job_id)
        if manifest_path.exists():
            manifest = self.load(job_id)
            if manifest.source_sha256 != source_hash:
                raise RuntimeError("job hash collision")
            return manifest
        manifest = JobManifest(
            schema_version=SCHEMA_VERSION,
            job_id=job_id,
            source_name=source.name,
            source_sha256=source_hash,
        )
        self.save(manifest)
        return manifest

    def load(self, job_id: str) -> JobManifest:
        path = self._manifest_path(job_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            stages = tuple(StageRecord(**item) for item in data.pop("stages", []))
            return JobManifest(**data, stages=stages)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid job manifest: {path}") from error

    def save(self, manifest: JobManifest) -> None:
        with FileLock(self._lock_path(manifest.job_id)):
            self._write_unlocked(manifest)

    def _write_unlocked(self, manifest: JobManifest) -> None:
        path = self._manifest_path(manifest.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(manifest)
        self._atomic_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")

    def update_stage(
        self,
        manifest: JobManifest,
        name: str,
        status: StageStatus,
        *,
        artifact: str | None = None,
        error: str | None = None,
    ) -> JobManifest:
        with FileLock(self._lock_path(manifest.job_id)):
            current = self.load(manifest.job_id)
            updated = current.with_stage(
                StageRecord(
                    name=name,
                    status=status,
                    artifact=artifact,
                    error=error,
                    updated_at=datetime.now(UTC).isoformat(),
                )
            )
            self._write_unlocked(updated)
        return updated

    def job_directory(self, manifest: JobManifest) -> Path:
        path = self._root / manifest.job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_path(self, manifest: JobManifest, relative_path: str) -> Path:
        root = self.job_directory(manifest).resolve()
        portable = PurePosixPath(relative_path)
        if portable.is_absolute() or "\\" in relative_path:
            raise ValueError("artifact path must be portable and relative")
        candidate = root.joinpath(*portable.parts).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("artifact path escapes the job directory")
        return candidate

    def _manifest_path(self, job_id: str) -> Path:
        return self._root / job_id / "manifest.json"

    def _lock_path(self, job_id: str) -> Path:
        return self._root / job_id / ".manifest.lock"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _atomic_text(path: Path, value: str) -> None:
        descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", text=True)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as target:
                target.write(value)
                target.flush()
                os.fsync(target.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
