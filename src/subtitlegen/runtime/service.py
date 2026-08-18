from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

from subtitlegen.asr.base import AsrBackend
from subtitlegen.asr.context import AsrContext
from subtitlegen.domain.models import Cue, Transcription, Word
from subtitlegen.pipeline import CueAssembler, SubtitleWriter
from subtitlegen.runtime.executor import StageExecutor
from subtitlegen.runtime.jobs import PortableJobStore
from subtitlegen.validation import is_valid_srt


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    status: Literal["generated", "resumed", "skipped"]
    output_path: Path
    job_id: str | None


class CueProcessor(Protocol):
    def process(self, cues: Iterable[Cue]) -> list[Cue]:
        """Apply deterministic text processing without changing cue timing."""


class RuntimeService:
    def __init__(
        self,
        backend: AsrBackend,
        cue_builder: CueAssembler,
        writer: SubtitleWriter,
        store: PortableJobStore,
        executor: StageExecutor,
        *,
        asr_key: str,
        output_key: str,
        context: AsrContext | None = None,
        cue_processor: CueProcessor | None = None,
    ) -> None:
        self._validate_key(asr_key)
        self._validate_key(output_key)
        self._backend = backend
        self._cue_builder = cue_builder
        self._writer = writer
        self._store = store
        self._executor = executor
        self._asr_key = asr_key
        self._output_key = output_key
        self._context = context
        self._cue_processor = cue_processor

    def process(
        self,
        media_path: Path,
        output_path: Path,
        *,
        language: str | None = None,
        overwrite: bool = False,
        refresh_stages: bool = False,
    ) -> RuntimeResult:
        manifest = self._store.create(media_path)
        if (
            is_valid_srt(output_path)
            and not overwrite
            and self._is_current_output(output_path, manifest.source_sha256)
        ):
            return RuntimeResult("skipped", output_path, manifest.job_id)
        initial_transcription = manifest.stage(f"transcribe-{self._asr_key}")

        def transcribe(job_directory: Path) -> Path:
            result = self._backend.transcribe(
                media_path,
                language=language,
                context=self._context,
            )
            artifact = job_directory / self._asr_key / "words.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_text(artifact, self._encode_transcription(result))
            return artifact

        manifest, words_path = self._executor.run(
            manifest,
            f"transcribe-{self._asr_key}",
            transcribe,
            validator=self._valid_transcription,
            force=refresh_stages,
        )

        def build_subtitle(job_directory: Path) -> Path:
            transcription = self._decode_transcription(words_path)
            cues = self._cue_builder.build(transcription.words)
            if self._cue_processor is not None:
                cues = self._cue_processor.process(cues)
            artifact = job_directory / self._output_key / "subtitle.srt"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._temporary_path(artifact.parent, ".srt.tmp")
            try:
                self._writer.write(cues, temporary)
                temporary.replace(artifact)
            finally:
                temporary.unlink(missing_ok=True)
            return artifact

        _, subtitle_path = self._executor.run(
            manifest,
            f"subtitle-{self._output_key}",
            build_subtitle,
            validator=is_valid_srt,
            force=refresh_stages,
        )
        self._atomic_copy(subtitle_path, output_path)
        self._write_output_metadata(output_path, manifest.source_sha256)
        status: Literal["generated", "resumed"] = (
            "resumed"
            if not refresh_stages
            and initial_transcription is not None
            and initial_transcription.status == "complete"
            else "generated"
        )
        return RuntimeResult(status, output_path, manifest.job_id)

    def set_cue_processor(self, cue_processor: CueProcessor | None) -> None:
        self._cue_processor = cue_processor

    def close(self) -> None:
        self._backend.close()

    def _is_current_output(self, output_path: Path, source_sha256: str) -> bool:
        try:
            data = json.loads(self._metadata_path(output_path).read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        return bool(
            data.get("schema_version") == 1
            and data.get("source_sha256") == source_sha256
        )

    def _write_output_metadata(self, output_path: Path, source_sha256: str) -> None:
        self._atomic_text(
            self._metadata_path(output_path),
            json.dumps(
                {
                    "schema_version": 1,
                    "source_sha256": source_sha256,
                    "output_key": self._output_key,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    @staticmethod
    def _metadata_path(output_path: Path) -> Path:
        return output_path.with_suffix(f"{output_path.suffix}.subtitlegen.json")

    @staticmethod
    def _validate_key(value: str) -> None:
        invalid_characters = (
            character not in "-_." for character in value if not character.isalnum()
        )
        if not value or any(invalid_characters):
            raise ValueError(
                "run key must contain only letters, numbers, dots, dashes, or underscores"
            )

    @staticmethod
    def _encode_transcription(transcription: Transcription) -> str:
        return json.dumps(
            {
                "schema_version": 1,
                "language": transcription.language,
                "duration": transcription.duration,
                "words": [asdict(word) for word in transcription.words],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    @staticmethod
    def _decode_transcription(path: Path) -> Transcription:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.pop("schema_version") != 1:
                raise ValueError("unsupported transcription schema")
            words = tuple(Word(**word) for word in data.pop("words"))
            return Transcription(words=words, **data)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid transcription artifact: {path}") from error

    @classmethod
    def _valid_transcription(cls, path: Path) -> bool:
        try:
            transcription = cls._decode_transcription(path)
        except (OSError, RuntimeError, UnicodeError):
            return False
        return bool(transcription.words)

    @staticmethod
    def _atomic_text(path: Path, value: str) -> None:
        temporary = RuntimeService._temporary_path(path.parent, f"{path.suffix}.tmp")
        try:
            temporary.write_text(value, encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = RuntimeService._temporary_path(
            destination.parent, f"{destination.suffix}.tmp"
        )
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _temporary_path(directory: Path, suffix: str) -> Path:
        descriptor, name = tempfile.mkstemp(dir=directory, prefix=".subtitlegen-", suffix=suffix)
        os.close(descriptor)
        return Path(name)
