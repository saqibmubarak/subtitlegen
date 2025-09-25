# transcriber.py

import whisper
from pathlib import Path
from typing import Dict, Any, Optional

# Global cache to hold the loaded Whisper model
_model_cache = {}


def load_whisper_model(model_identifier: str, device: str):
    """
    Loads and caches the Whisper model based on identifier and device.

    Requirement: Uses the LLM (Whisper).
    """
    if model_identifier not in _model_cache:
        print(f"Loading Whisper model: {model_identifier} on device: {device}...")
        # whisper.load_model handles downloading and setting up PyTorch device (cuda/cpu)
        try:
            model = whisper.load_model(model_identifier, device=device)
            _model_cache[model_identifier] = model
            print("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Whisper model '{model_identifier}' on device '{device}'. Check model name and PyTorch/CUDA setup.") from e

    return _model_cache[model_identifier]


def transcribe_video(
        video_path: Path,
        model_identifier: str,
        device: str,
        language: Optional[str]
) -> Dict[str, Any]:
    """Transcribes a single video file and returns the full result dictionary."""

    model = load_whisper_model(model_identifier, device)

    print(f"-> Starting transcription for: {video_path.name}")

    # Transcribe the audio stream extracted from the video file
    result = model.transcribe(
        str(video_path),
        language=language,
        # Default options for segments should yield segments with timestamps
    )

    print(f"-> Transcription complete. Detected language: {result.get('language', 'N/A')}")

    return result