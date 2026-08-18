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
from subtitlegen.export.ass import AssWriter
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
from subtitlegen.visual.detection import (
    FallbackTextDetector,
    OpenCvDbNetDetector,
    PaddleOcrDetector,
)
from subtitlegen.visual.merger import SubtitleMerger
from subtitlegen.visual.ocr import MangaOcrEngine
from subtitlegen.visual.pipeline import VisualTextPipeline
from subtitlegen.visual.proposals import TemporalDifferenceProposer
from subtitlegen.visual.sampler import FrameSampler
from subtitlegen.visual.service import MultimodalSubtitleService
from subtitlegen.visual.settings import VisualPipelineSettings
from subtitlegen.visual.tracker import VisualEventTracker
from subtitlegen.visual.translation import NllbLocalTranslator

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


def _visual_service(
    profile: SeriesProfile | None,
    detector_model: Path | None,
    cache_dir: Path,
    frames_per_second: float,
    minimum_japanese_characters: int = 5,
) -> MultimodalSubtitleService:
    visual_settings = VisualPipelineSettings(
        frames_per_second=frames_per_second,
        minimum_japanese_characters=minimum_japanese_characters,
    )
    paddle = PaddleOcrDetector()
    detector = (
        FallbackTextDetector(OpenCvDbNetDetector(detector_model), paddle)
        if detector_model is not None
        else paddle
    )
    pipeline = VisualTextPipeline(
        FrameSampler(
            frames_per_second=visual_settings.frames_per_second,
            scene_threshold=visual_settings.scene_threshold,
        ),
        detector,
        MangaOcrEngine(),
        NllbLocalTranslator(
            profile=profile,
            device="cuda" if DeviceCapabilities.detect().cuda_devices else "cpu",
        ),
        VisualEventTracker(
            max_gap_seconds=visual_settings.tracker_max_gap_seconds,
            frame_interval_seconds=visual_settings.frame_interval_seconds,
            min_observations=visual_settings.tracker_minimum_observations,
            box_iou_threshold=visual_settings.tracker_box_iou_threshold,
            text_similarity_threshold=visual_settings.tracker_text_similarity_threshold,
            hash_distance_threshold=visual_settings.tracker_hash_distance_threshold,
        ),
        region_proposer=TemporalDifferenceProposer(
            difference_threshold=visual_settings.proposal_difference_threshold,
            minimum_changed_area_ratio=visual_settings.proposal_minimum_area_ratio,
            maximum_changed_area_ratio=visual_settings.proposal_maximum_area_ratio,
            padding_ratio=visual_settings.proposal_padding_ratio,
            hold_frames=visual_settings.proposal_hold_frames,
            full_frame_hold_frames=visual_settings.proposal_full_frame_hold_frames,
            analysis_width=visual_settings.proposal_analysis_width,
            maximum_regions=visual_settings.proposal_maximum_regions,
        ),
        detector_input_size=visual_settings.detector_input_size,
        minimum_japanese_characters=visual_settings.minimum_japanese_characters,
        minimum_box_area_ratio=visual_settings.minimum_box_area_ratio,
        minimum_vertical_center_ratio=visual_settings.minimum_vertical_center_ratio,
    )
    detector_identity = "paddle-ppocrv5-mobile"
    if detector_model is not None:
        with detector_model.open("rb") as model_file:
            detector_identity = hashlib.file_digest(model_file, "sha256").hexdigest()
    visual_data = (
        detector_identity,
        "manga-ocr-0.1.16",
        NllbLocalTranslator.DEFAULT_MODEL,
        profile,
        visual_settings.cache_identity(),
    )
    visual_key = "visual-" + hashlib.sha256(repr(visual_data).encode()).hexdigest()[:12]
    store = PortableJobStore(cache_dir / "jobs")
    return MultimodalSubtitleService(
        pipeline,
        SubtitleMerger(),
        AssWriter(),
        store=store,
        executor=StageExecutor(store, GpuResourceToken()),
        visual_key=visual_key,
    )


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
    visual_text: Annotated[bool, typer.Option("--visual-text")] = False,
    visual_fps: Annotated[float, typer.Option("--visual-fps")] = 1.5,
    visual_min_japanese_characters: Annotated[
        int,
        typer.Option("--visual-min-japanese-characters", min=1),
    ] = 5,
    detector_model: Annotated[Path | None, typer.Option("--detector-model")] = None,
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
    generated: list[tuple[Path, Path]] = []
    input_root = input_path.resolve()
    try:
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
                generated.append((video, output))
                logger.info("%s: %s using %s", result.status, output, selected)
            except Exception:
                failures += 1
                logger.exception("failed: %s", video)
    finally:
        close = getattr(service, "close", None)
        if close is not None:
            close()

    if visual_text:
        multimodal = _visual_service(
            series_profile,
            detector_model,
            cache_dir,
            visual_fps,
                visual_min_japanese_characters,
        )
        try:
            for video, dialogue_output in generated:
                try:
                    visual_result = multimodal.process(
                        video,
                        dialogue_output,
                        dialogue_output.with_suffix(".ass"),
                    )
                    logger.info(
                        "generated: %s with %d visual events",
                        visual_result.output_path,
                        visual_result.visual_events,
                    )
                except Exception:
                    failures += 1
                    logger.exception("visual text failed: %s", video)
        finally:
            multimodal.close()
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
    try:
        result = service.process(
            media_path,
            output,
            language=settings.asr.language,
            overwrite=True,
            refresh_stages=True,
        )
    finally:
        service.close()
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
