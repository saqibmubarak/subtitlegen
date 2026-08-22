from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Annotated

import typer

from subtitlegen.cues.builder import CueBuilder
from subtitlegen.errors import SubtitleWriteError
from subtitlegen.export.ass import AssWriter
from subtitlegen.export.srt import SrtWriter
from subtitlegen.logging import configure_logging
from subtitlegen.media import discover_media, media_duration
from subtitlegen.profiles.builder import AutomaticProfileBuilder
from subtitlegen.profiles.correction import (
    ConfidenceGatedCorrector,
    ConservativeLocalCorrector,
)
from subtitlegen.profiles.cue_processor import ProfileCueProcessor
from subtitlegen.profiles.extraction import LocalTranscriptExtractor
from subtitlegen.profiles.models import SeriesProfile
from subtitlegen.profiles.normalizer import GlossaryNormalizer
from subtitlegen.profiles.repository import ProfileRepository
from subtitlegen.profiles.resolver import ProfileResolver, ResolvedProfile
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
    disable_paddle_onednn,
)
from subtitlegen.visual.merger import SubtitleMerger
from subtitlegen.visual.ocr import MangaOcrEngine, PaddleTextRecognizer
from subtitlegen.visual.pipeline import VisualTextPipeline
from subtitlegen.visual.presence import JapaneseCharacterScanner
from subtitlegen.visual.proposals import TemporalDifferenceProposer
from subtitlegen.visual.sampler import AdaptiveVisualSampler
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
    cue_processor = (
        _cue_processor(profile, local_correction=local_correction)
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
        profile.profile_id if profile is not None else None,
        arc,
        episode,
        "asr-decode-v2",
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
    logger.info(
        "preset %s selected %s %s",
        resolved.name,
        resolved.backend,
        resolved.settings.model,
    )
    return replace(settings, asr=resolved.settings), resolved.backend


def _shipped_repository(profiles_dir: Path | None) -> ProfileRepository | None:
    if profiles_dir is not None:
        return ProfileRepository(profiles_dir)
    try:
        return ProfileRepository.default()
    except FileNotFoundError:
        return None


def _resolve_profile(
    paths: tuple[Path, ...],
    cache_dir: Path,
    profile: str | None,
    profiles_dir: Path | None,
    *,
    auto: bool,
) -> ResolvedProfile:
    shipped = _shipped_repository(profiles_dir)
    cache = ProfileRepository(cache_dir / "profiles")
    resolved = ProfileResolver(
        cache=cache,
        shipped=shipped,
    ).resolve(
        paths,
        explicit_id=profile,
        auto=auto,
        explicit_repository=shipped,
    )
    return _expand_glossary(resolved, cache)


def _glossary_enrichment_enabled() -> bool:
    return os.environ.get("SUBTITLEGEN_ENRICH_GLOSSARY", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _expand_glossary(
    resolved: ResolvedProfile,
    cache: ProfileRepository,
) -> ResolvedProfile:
    if resolved.profile is None or not _glossary_enrichment_enabled():
        return resolved
    if len(resolved.profile.terms) >= 80:
        return resolved
    try:
        updated = AutomaticProfileBuilder().enrich(resolved.profile)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        logger.warning("glossary enrichment failed: %s", error)
        return resolved
    if len(updated.terms) <= len(resolved.profile.terms):
        return resolved
    cache.save(updated)
    logger.info(
        "expanded glossary for %s from %d to %d terms",
        updated.title,
        len(resolved.profile.terms),
        len(updated.terms),
    )
    return ResolvedProfile(
        updated,
        resolved.identity,
        f"{resolved.source}+wikipedia",
        resolved.enable_visual,
    )


def _cue_processor(
    profile: SeriesProfile,
    *,
    local_correction: bool,
) -> ProfileCueProcessor:
    normalizer = GlossaryNormalizer()
    corrector = (
        ConfidenceGatedCorrector(normalizer, ConservativeLocalCorrector())
        if local_correction
        else None
    )
    return ProfileCueProcessor(profile, normalizer, corrector)


def _enrich_profile(
    profile: SeriesProfile,
    output_path: Path,
    cache: ProfileRepository,
    service: RuntimeService,
    *,
    local_correction: bool,
) -> SeriesProfile:
    cues = parse_srt(output_path)
    updated = LocalTranscriptExtractor().enrich(profile, cues)
    if updated.terms == profile.terms:
        return profile
    cache.save(updated)
    processor = _cue_processor(updated, local_correction=local_correction)
    SrtWriter().write(processor.process(cues), output_path)
    service.set_cue_processor(processor)
    return updated


def _visual_service(
    profile: SeriesProfile | None,
    detector_model: Path | None,
    cache_dir: Path,
    frames_per_second: float,
    minimum_japanese_characters: int = 5,
    probe_interval_seconds: float = 4.0,
    refine_window_seconds: float = 12.0,
) -> MultimodalSubtitleService:
    visual_settings = VisualPipelineSettings(
        frames_per_second=frames_per_second,
        probe_interval_seconds=probe_interval_seconds,
        refine_window_seconds=refine_window_seconds,
        minimum_japanese_characters=minimum_japanese_characters,
    )
    disable_paddle_onednn()
    paddle = PaddleOcrDetector()
    detector = (
        FallbackTextDetector(OpenCvDbNetDetector(detector_model), paddle)
        if detector_model is not None
        else paddle
    )
    recognizer = PaddleTextRecognizer()
    scanner = JapaneseCharacterScanner(paddle, recognizer)
    pipeline = VisualTextPipeline(
        AdaptiveVisualSampler(
            scanner,
            frames_per_second=visual_settings.frames_per_second,
            probe_interval_seconds=visual_settings.probe_interval_seconds,
            refine_window_seconds=visual_settings.refine_window_seconds,
            refine_interval_seconds=visual_settings.refine_interval_seconds,
            scene_threshold=visual_settings.scene_threshold,
            skip_nonref_frames=visual_settings.skip_nonref_frames,
        ),
        detector,
        MangaOcrEngine(),
        NllbLocalTranslator(
            profile=profile,
            device="cuda" if DeviceCapabilities.detect().cuda_devices else "cpu",
        ),
        VisualEventTracker(
            max_gap_seconds=visual_settings.tracker_max_gap_seconds,
            frame_interval_seconds=visual_settings.refine_interval_seconds,
            min_observations=visual_settings.tracker_minimum_observations,
            box_iou_threshold=visual_settings.tracker_box_iou_threshold,
            text_similarity_threshold=visual_settings.tracker_text_similarity_threshold,
            hash_distance_threshold=visual_settings.tracker_hash_distance_threshold,
        ),
        line_ocr=recognizer,
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
        "title-scan-v7",
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
    config: Annotated[Path, typer.Option("--config", envvar="SUBTITLEGEN_CONFIG")] = Path(
        "config.ini"
    ),
    backend: Annotated[str, typer.Option("--backend", envvar="SUBTITLEGEN_BACKEND")] = "auto",
    preset: Annotated[
        str | None,
        typer.Option("--preset", envvar="SUBTITLEGEN_PRESET"),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", envvar="SUBTITLEGEN_OUTPUT_DIR"),
    ] = None,
    cache_dir: Annotated[
        Path,
        typer.Option("--cache-dir", envvar="SUBTITLEGEN_CACHE_DIR"),
    ] = Path(".subtitlegen"),
    profile: Annotated[
        str | None,
        typer.Option("--profile", envvar="SUBTITLEGEN_PROFILE"),
    ] = None,
    profiles_dir: Annotated[Path | None, typer.Option("--profiles-dir")] = None,
    arc: Annotated[str | None, typer.Option("--arc", envvar="SUBTITLEGEN_ARC")] = None,
    episode: Annotated[str | None, typer.Option("--episode")] = None,
    auto_profile: Annotated[bool, typer.Option("--auto-profile/--no-auto-profile")] = True,
    local_correction: Annotated[
        bool,
        typer.Option("--local-correction/--no-local-correction"),
    ] = True,
    visual_text: Annotated[
        bool | None,
        typer.Option("--visual-text/--no-visual-text"),
    ] = None,
    visual_fps: Annotated[
        float,
        typer.Option("--visual-fps", envvar="SUBTITLEGEN_VISUAL_FPS"),
    ] = 1.5,
    visual_probe_seconds: Annotated[
        float,
        typer.Option("--visual-probe-seconds", envvar="SUBTITLEGEN_VISUAL_PROBE_SECONDS"),
    ] = 4.0,
    visual_refine_seconds: Annotated[
        float,
        typer.Option("--visual-refine-seconds", envvar="SUBTITLEGEN_VISUAL_REFINE_SECONDS"),
    ] = 12.0,
    visual_min_japanese_characters: Annotated[
        int,
        typer.Option("--visual-min-japanese-characters", min=1),
    ] = 5,
    detector_model: Annotated[Path | None, typer.Option("--detector-model")] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", envvar="SUBTITLEGEN_OVERWRITE"),
    ] = False,
    reuse_srt: Annotated[
        bool,
        typer.Option("--reuse-srt/--transcribe", envvar="SUBTITLEGEN_REUSE_SRT"),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Generate SRT files recursively, resuming valid cached stages."""
    configure_logging(verbose)
    settings = SettingsLoader().load(config)
    videos = discover_media(input_path, settings.video_extensions)
    if not videos:
        raise typer.BadParameter("no supported media files were found")
    resolved = _resolve_profile(
        (input_path, *videos[:3]),
        cache_dir,
        profile,
        profiles_dir,
        auto=auto_profile,
    )
    series_profile = resolved.profile
    scoped = resolved.identity if not input_path.is_dir() else None
    selected_arc = arc or (scoped.arc if scoped is not None else None)
    selected_episode = episode or (scoped.episode if scoped is not None else None)
    use_visual = True if visual_text is None else visual_text
    if series_profile is not None:
        logger.info(
            "glossary %s (%s) has %d terms",
            series_profile.title,
            resolved.source,
            len(series_profile.terms),
        )
    logger.info(
        "on-screen text extraction %s",
        "enabled" if use_visual else "disabled",
    )

    failures = 0
    generated: list[tuple[Path, Path]] = []
    input_root = input_path.resolve()

    def output_for(video: Path) -> Path:
        if output_dir is None:
            return video.with_suffix(".srt")
        if input_root.is_dir():
            return output_dir / video.relative_to(input_root).with_suffix(".srt")
        return output_dir / f"{video.stem}.srt"

    jobs = [(video, output_for(video)) for video in videos]

    if reuse_srt:
        logger.info("reusing existing dialogue SRT; ASR is disabled")
        for video, output in jobs:
            if is_valid_srt(output):
                generated.append((video, output))
                logger.info("reused: %s", output)
            else:
                failures += 1
                logger.error("no dialogue SRT for %s", video)
    else:
        settings, backend = _resolve_runtime(settings, backend, preset)
        service, selected = _service(
            settings,
            backend,
            cache_dir,
            series_profile,
            arc=selected_arc,
            episode=selected_episode,
            local_correction=local_correction,
        )

        def needed_from(start: int) -> list[Path]:
            return [
                video
                for video, output in jobs[start:]
                if overwrite or not is_valid_srt(output)
            ]

        prefetch = getattr(service, "prefetch_audio", None)
        if prefetch is not None:
            for video in needed_from(0)[:2]:
                prefetch(video)
        enrich_outputs: list[Path] = []
        try:
            for index, (video, output) in enumerate(jobs):
                if prefetch is not None:
                    for video_ahead in needed_from(index + 1)[:2]:
                        prefetch(video_ahead)
                try:
                    result = service.process(
                        video,
                        output,
                        language=settings.asr.language,
                        overwrite=overwrite,
                    )
                    generated.append((video, output))
                    if result.status == "skipped":
                        logger.info(
                            "skipped: %s already exists; pass --overwrite to regenerate",
                            output,
                        )
                    else:
                        enrich_outputs.append(output)
                        logger.info("%s: %s using %s", result.status, output, selected)
                except SubtitleWriteError as error:
                    failures += 1
                    logger.exception("failed: %s", error.path)
                except Exception:
                    failures += 1
                    logger.exception("failed: %s", video)
            flush = getattr(service, "flush_writes", None)
            if flush is not None:
                flush()
            if series_profile is not None and auto_profile:
                for output in enrich_outputs:
                    series_profile = _enrich_profile(
                        series_profile,
                        output,
                        ProfileRepository(cache_dir / "profiles"),
                        service,
                        local_correction=local_correction,
                    )
        finally:
            close = getattr(service, "close", None)
            if close is not None:
                close()

    if use_visual:
        multimodal = _visual_service(
            series_profile,
            detector_model,
            cache_dir,
            visual_fps,
            visual_min_japanese_characters,
            visual_probe_seconds,
            visual_refine_seconds,
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
    auto_profile: Annotated[bool, typer.Option("--auto-profile/--no-auto-profile")] = True,
    local_correction: Annotated[
        bool,
        typer.Option("--local-correction/--no-local-correction"),
    ] = True,
) -> None:
    """Measure one media file using the selected local backend."""
    settings = SettingsLoader().load(config)
    settings, backend = _resolve_runtime(settings, backend, preset)
    resolved = _resolve_profile(
        (media_path,),
        cache_dir,
        profile,
        profiles_dir,
        auto=auto_profile,
    )
    series_profile = resolved.profile
    selected_arc = arc or (resolved.identity.arc if resolved.identity is not None else None)
    selected_episode = episode or (
        resolved.identity.episode if resolved.identity is not None else None
    )
    service, selected = _service(
        settings,
        backend,
        cache_dir,
        series_profile,
        arc=selected_arc,
        episode=selected_episode,
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
