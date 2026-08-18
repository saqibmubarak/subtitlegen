from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Annotated

import typer

from subtitlegen.cues.builder import CueBuilder
from subtitlegen.export.srt import SrtWriter
from subtitlegen.logging import configure_logging
from subtitlegen.media import discover_media, media_duration
from subtitlegen.profiles.correction import (
    ConfidenceGatedCorrector,
    ConservativeLocalCorrector,
)
from subtitlegen.profiles.cue_processor import ProfileCueProcessor
from subtitlegen.profiles.models import SeriesProfile
from subtitlegen.profiles.normalizer import GlossaryNormalizer
from subtitlegen.profiles.repository import ProfileRepository
from subtitlegen.profiles.selector import ContextSelector
from subtitlegen.runtime.capabilities import DeviceCapabilities
from subtitlegen.runtime.executor import GpuResourceToken, StageExecutor
from subtitlegen.runtime.factory import BackendFactory
from subtitlegen.runtime.jobs import PortableJobStore
from subtitlegen.runtime.presets import PresetResolver
from subtitlegen.runtime.service import RuntimeService
from subtitlegen.settings import AppSettings, SettingsLoader
from subtitlegen.validation import analyze_cues, is_valid_srt, parse_srt

app = typer.Typer(no_args_is_help=True, help="Generate portable, synchronized subtitles.")
logger = logging.getLogger(__name__)


def _service(
    settings: AppSettings,
    backend_name: str,
    cache_dir: Path,
    profile: SeriesProfile | None = None,
    *,
    arc: str | None = None,
    episode: str | None = None,
    local_correction: bool = False,
) -> tuple[RuntimeService, str]:
    capabilities = DeviceCapabilities.detect()
    factory = BackendFactory(capabilities)
    selected = factory.select(backend_name)
    backend = factory.create(selected, settings.asr)
    context = (
        ContextSelector().select(profile, arc=arc, episode=episode)
        if profile is not None
        else None
    )
    normalizer = GlossaryNormalizer()
    gated_corrector = (
        ConfidenceGatedCorrector(normalizer, ConservativeLocalCorrector())
        if profile is not None and local_correction
        else None
    )
    cue_processor = (
        ProfileCueProcessor(profile, normalizer, gated_corrector)
        if profile is not None
        else None
    )
    asr_data = (
        selected,
        settings.asr.model,
        settings.asr.device,
        settings.asr.compute_type,
        settings.asr.language,
        settings.asr.beam_size,
        settings.asr.whisperx_batch_size,
        settings.asr.vad,
        context,
    )
    asr_hash = hashlib.sha256(repr(asr_data).encode()).hexdigest()[:12]
    asr_key = f"{selected}-{asr_hash}"
    output_hash = hashlib.sha256(
        repr((asr_key, settings.cues, profile, local_correction)).encode()
    ).hexdigest()[:12]
    output_key = f"srt-{output_hash}"
    store = PortableJobStore(cache_dir / "jobs")
    executor = StageExecutor(store, GpuResourceToken())
    return (
        RuntimeService(
            backend,
            CueBuilder(settings.cues),
            SrtWriter(),
            store,
            executor,
            asr_key=asr_key,
            output_key=output_key,
            context=context,
            cue_processor=cue_processor,
        ),
        selected,
    )


def _resolve_runtime(
    settings: AppSettings,
    backend_name: str,
    preset: str | None,
) -> tuple[AppSettings, str]:
    if preset is None:
        return settings, backend_name
    if backend_name != "auto":
        raise ValueError("--preset cannot be combined with an explicit --backend")
    resolved = PresetResolver().resolve(
        preset,
        DeviceCapabilities.detect(),
        settings.asr,
    )
    return replace(settings, asr=resolved.settings), resolved.backend


@app.command()
def generate(
    input_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    config: Annotated[Path, typer.Option("--config")] = Path("config.ini"),
    backend: Annotated[str, typer.Option("--backend")] = "auto",
    preset: Annotated[str | None, typer.Option("--preset")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(".subtitlegen"),
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    profiles_dir: Annotated[Path | None, typer.Option("--profiles-dir")] = None,
    arc: Annotated[str | None, typer.Option("--arc")] = None,
    episode: Annotated[str | None, typer.Option("--episode")] = None,
    local_correction: Annotated[bool, typer.Option("--local-correction")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Generate SRT files recursively, resuming valid cached stages."""
    configure_logging(verbose)
    settings = SettingsLoader().load(config)
    settings, backend = _resolve_runtime(settings, backend, preset)
    series_profile = None
    if profile:
        profile_repository = (
            ProfileRepository(profiles_dir)
            if profiles_dir is not None
            else ProfileRepository.default()
        )
        series_profile = profile_repository.load(profile)
    service, selected = _service(
        settings,
        backend,
        cache_dir,
        series_profile,
        arc=arc,
        episode=episode,
        local_correction=local_correction,
    )
    videos = discover_media(input_path, settings.video_extensions)
    if not videos:
        raise typer.BadParameter("no supported media files were found")

    failures = 0
    input_root = input_path.resolve()
    for video in videos:
        if output_dir is None:
            output = video.with_suffix(".srt")
        elif input_root.is_dir():
            output = output_dir / video.relative_to(input_root).with_suffix(".srt")
        else:
            output = output_dir / f"{video.stem}.srt"
        try:
            result = service.process(
                video,
                output,
                language=settings.asr.language,
                overwrite=overwrite,
            )
            logger.info("%s: %s using %s", result.status, output, selected)
        except Exception:
            failures += 1
            logger.exception("failed: %s", video)
    if failures:
        raise typer.Exit(code=1)


@app.command()
def validate(
    subtitle_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Validate an SRT file and print deterministic timing metrics."""
    if not is_valid_srt(subtitle_path):
        typer.echo(json.dumps({"valid": False, "path": str(subtitle_path)}))
        raise typer.Exit(code=1)
    report = analyze_cues(parse_srt(subtitle_path))
    typer.echo(json.dumps({"valid": True, **asdict(report)}, sort_keys=True))


@app.command()
def benchmark(
    media_path: Annotated[Path, typer.Argument(exists=True, readable=True, dir_okay=False)],
    config: Annotated[Path, typer.Option("--config")] = Path("config.ini"),
    backend: Annotated[str, typer.Option("--backend")] = "auto",
    preset: Annotated[str | None, typer.Option("--preset")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(".subtitlegen"),
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    profiles_dir: Annotated[Path | None, typer.Option("--profiles-dir")] = None,
    arc: Annotated[str | None, typer.Option("--arc")] = None,
    episode: Annotated[str | None, typer.Option("--episode")] = None,
    local_correction: Annotated[bool, typer.Option("--local-correction")] = False,
) -> None:
    """Measure one media file using the selected local backend."""
    settings = SettingsLoader().load(config)
    settings, backend = _resolve_runtime(settings, backend, preset)
    series_profile = None
    if profile:
        profile_repository = (
            ProfileRepository(profiles_dir)
            if profiles_dir is not None
            else ProfileRepository.default()
        )
        series_profile = profile_repository.load(profile)
    service, selected = _service(
        settings,
        backend,
        cache_dir,
        series_profile,
        arc=arc,
        episode=episode,
        local_correction=local_correction,
    )
    output = cache_dir / "benchmarks" / f"{media_path.stem}-{selected}.srt"
    started = time.perf_counter()
    result = service.process(
        media_path,
        output,
        language=settings.asr.language,
        overwrite=True,
        refresh_stages=True,
    )
    elapsed = time.perf_counter() - started
    duration = media_duration(media_path)
    report = analyze_cues(parse_srt(output))
    typer.echo(
        json.dumps(
            {
                "backend": selected,
                "elapsed_seconds": elapsed,
                "media_seconds": duration,
                "realtime_factor": elapsed / duration if duration else None,
                "status": result.status,
                "timing": asdict(report),
            },
            sort_keys=True,
        )
    )
