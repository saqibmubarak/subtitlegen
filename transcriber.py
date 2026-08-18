import whisper
from pathlib import Path
from typing import Dict, Any, Optional

from whisper import Whisper, DecodingOptions

# Global cache is crucial for multiprocessing performance
_model_cache = {}


def load_whisper_model(model_identifier: str, device: str, compute_type: str) -> Whisper:
    """
    Loads and caches the Faster Whisper model based on identifier, device, and compute type.
    This runs only once per process, solving the repeated loading bottleneck.
    """
    cache_key = f"{model_identifier}-{device}-{compute_type}"

    if cache_key not in _model_cache:
        print(
            f"Loading model: {model_identifier} on device: {device} with compute type: {compute_type}...")
        try:
            model = whisper.load_model(model_identifier, device=device, in_memory=True, download_root="/cache")
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
) -> Dict[str, Whisper]:
    """Transcribes a single video file and returns the result in the expected dictionary format."""

    model = load_whisper_model(model_identifier, device, compute_type)

    print(f"-> Starting transcription for: {video_path.name}")
    print(video_path)
    # 1. Perform Transcription
    result = model.transcribe(
        str(video_path),
        language=language,
        verbose=False,
        fp16=False
    )
    # print(result)
    # # 2. Extract segments from the result
    # segments = []
    # for segment in result["segments"]:
    #     segments.append({
    #         'start': segment['start'],
    #         'end': segment['end'],
    #         'text': segment['text']
    #     })

    print(f"-> Transcription complete. Detected language: {result.get('language', 'N/A')}")

    return result