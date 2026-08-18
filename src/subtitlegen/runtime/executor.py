from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import BoundedSemaphore

from subtitlegen.runtime.jobs import FileLock, JobManifest, PortableJobStore

StageAction = Callable[[Path], Path]
ArtifactValidator = Callable[[Path], bool]


class GpuResourceToken:
    """Bound model-heavy stages to one concurrent GPU owner by default."""

    def __init__(self, capacity: int = 1) -> None:
        if capacity <= 0:
            raise ValueError("GPU resource capacity must be positive")
        self._semaphore = BoundedSemaphore(capacity)

    @contextmanager
    def acquire(self) -> Iterator[None]:
        self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()


class StageExecutor:
    def __init__(self, store: PortableJobStore, resource: GpuResourceToken | None = None) -> None:
        self._store = store
        self._resource = resource

    def run(
        self,
        manifest: JobManifest,
        stage_name: str,
        action: StageAction,
        *,
        validator: ArtifactValidator | None = None,
        force: bool = False,
    ) -> tuple[JobManifest, Path]:
        lock_name = hashlib.sha256(stage_name.encode()).hexdigest()[:16]
        lock_path = self._store.job_directory(manifest) / f".stage-{lock_name}.lock"
        with FileLock(lock_path, timeout_seconds=7_200):
            current_manifest = self._store.load(manifest.job_id)
            return self._run_locked(
                current_manifest,
                stage_name,
                action,
                validator=validator,
                force=force,
            )

    def _run_locked(
        self,
        manifest: JobManifest,
        stage_name: str,
        action: StageAction,
        *,
        validator: ArtifactValidator | None,
        force: bool,
    ) -> tuple[JobManifest, Path]:
        current = manifest.stage(stage_name)
        if (
            not force
            and current is not None
            and current.status == "complete"
            and current.artifact
        ):
            artifact = self._store.artifact_path(manifest, current.artifact)
            if artifact.is_file() and (validator is None or validator(artifact)):
                return manifest, artifact

        manifest = self._store.update_stage(manifest, stage_name, "running")
        try:
            if self._resource is None:
                artifact = action(self._store.job_directory(manifest))
            else:
                with self._resource.acquire():
                    artifact = action(self._store.job_directory(manifest))
            root = self._store.job_directory(manifest).resolve()
            artifact = artifact.resolve()
            if (
                not artifact.is_file()
                or not artifact.is_relative_to(root)
                or (validator is not None and not validator(artifact))
            ):
                raise RuntimeError("stage did not create an artifact inside its job directory")
            relative = artifact.relative_to(root).as_posix()
            manifest = self._store.update_stage(
                manifest,
                stage_name,
                "complete",
                artifact=relative,
            )
            return manifest, artifact
        except (KeyboardInterrupt, SystemExit):
            self._store.update_stage(
                manifest,
                stage_name,
                "cancelled",
                error="cancelled by user",
            )
            raise
        except Exception as error:
            self._store.update_stage(manifest, stage_name, "failed", error=str(error))
            raise
