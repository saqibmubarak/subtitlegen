from faster_whisper import WhisperModel
from pathlib import Path
from typing import Dict, Any, Optional

# Global cache is crucial for multiprocessing performance
_model_cache = {}


def load_whisper_model(model_identifier: str, device: str, compute_type: str):
    """
    Loads and caches the Faster Whisper model based on identifier, device, and compute type.
    This runs only once per process, solving the repeated loading bottleneck.
    """
    cache_key = f"{model_identifier}-{device}-{compute_type}"

    if cache_key not in _model_cache:
        print(
            f"Loading Faster Whisper model: {model_identifier} on device: {device} with compute type: {compute_type}...")
        try:
            model = WhisperModel(
                model_identifier,
                device=device,
                compute_type=compute_type
            )
            _model_cache[cache_key] = model
            print("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Whisper model '{model_identifier}' on device '{device}'. Error: {e}") from e

    return _model_cache[cache_key]


def transcribe_video(
        video_path: Path,
        model_identifier: str,
        device: str,
        language: Optional[str],
        compute_type: str
) -> Dict[str, Any]:
    """Transcribes a single video file and returns the result in the expected dictionary format."""

    model = load_whisper_model(model_identifier, device, compute_type)

    print(f"-> Starting transcription for: {video_path.name}")

    # 1. Perform Transcription
    segments_generator, info = model.transcribe(
        str(video_path),
        language=language,
        vad_filter=True
    )

    # 2. Convert generator output to the list format expected by utils.py
    segments = []
    for segment in segments_generator:
        segments.append({
            'start': segment.start,
            'end': segment.end,
            'text': segment.text
        })

    detected_language = info.language if info else "N/A"
    print(f"-> Transcription complete. Detected language: {detected_language}")

    return {
        'segments': segments,
        'language': detected_language
    }