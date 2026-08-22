from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from threading import BoundedSemaphore, Thread
from typing import TypeVar

from subtitlegen.runtime.jobs import FileLock, JobManifest, PortableJobStore

StageAction = Callable[[Path], Path]
ArtifactValidator = Callable[[Path], bool]
T = TypeVar("T")

DECODE_PREFETCH = 12


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
        use_resource: bool = True,
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
                use_resource=use_resource,
            )

    def _run_locked(
        self,
        manifest: JobManifest,
        stage_name: str,
        action: StageAction,
        *,
        validator: ArtifactValidator | None,
        force: bool,
        use_resource: bool = True,
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
            if self._resource is None or not use_resource:
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


class PrefetchInferExecutor:
    """Bounded decode prefetch into a single infer thread."""

    def __init__(
        self,
        *,
        prefetch: int = DECODE_PREFETCH,
        thread_name: str = "title-infer",
    ) -> None:
        if prefetch < 1:
            raise ValueError("decode prefetch must be at least 1")
        self._prefetch = prefetch
        self._thread_name = thread_name

    def run(
        self,
        items: Iterable[T],
        process_batch: Callable[[list[T]], None],
        *,
        batch_size: int,
    ) -> int:
        if batch_size < 1:
            raise ValueError("infer batch size must be at least 1")
        queue: Queue[T | None] = Queue(maxsize=self._prefetch)
        errors: list[BaseException] = []
        counted = 0

        def infer() -> None:
            batch: list[T] = []
            finished = False
            try:
                while True:
                    item = queue.get()
                    if item is None:
                        finished = True
                        if batch:
                            process_batch(batch)
                        return
                    batch.append(item)
                    if len(batch) >= batch_size:
                        process_batch(batch)
                        batch.clear()
            except BaseException as error:
                errors.append(error)
            finally:
                if not finished:
                    while True:
                        item = queue.get()
                        if item is None:
                            break

        thread = Thread(target=infer, name=self._thread_name, daemon=True)
        thread.start()
        try:
            for item in items:
                if errors:
                    break
                queue.put(item)
                counted += 1
        finally:
            queue.put(None)
            thread.join()
        if errors:
            raise errors[0]
        return counted
