# main.py

import argparse
import sys
from pathlib import Path
import torch

# Import project modules
from config import load_config
from transcriber import transcribe_video
from utils import find_video_files, write_srt


def main():
    """Main execution function for the CLI application."""

    # --- 1. Argument Parsing (Requirement: CLI) ---
    parser = argparse.ArgumentParser(
        description="A command-line tool to generate SRT subtitles for video files using OpenAI Whisper.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a single video file OR a directory for recursive batch processing."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.ini",
        help="Path to the configuration file (default: config.ini)."
    )

    args = parser.parse_args()

    # --- 2. Load Configuration ---
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Fatal Error: Failed to load configuration. {e}")
        sys.exit(1)

    input_path = Path(args.input_path).resolve()

    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)

    # Check for GPU availability and update device setting
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA device not found. Transcription will use CPU, which is significantly slower.")
        device = 'cpu'

    # --- 3. Find Files (Requirement: Batch Processing) ---
    print(f"--- Configuration ---")
    print(f"Model: {config['model_identifier']}, Device: {device}, Language: {config['language'] or 'Auto'}")
    print(f"Scanning for supported files in: {input_path}")

    video_files = find_video_files(input_path, config['video_extensions'])

    if not video_files:
        print(f"No supported video files found in {input_path} (or its subdirectories).")
        return

    print(f"Found {len(video_files)} file(s) for processing.")

    # --- 4. Process Files ---
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")

        try:
            # 4a. Transcribe
            result = transcribe_video(
                video_file,
                config['model_identifier'],
                device,
                config['language']
            )

            # 4b. Define Output Path (Requirement: Same path, same basename, .srt extension)
            srt_output_path = video_file.with_suffix('.srt')

            # 4c. Write SRT
            write_srt(result['segments'], srt_output_path)

            print(f"SUCCESS: Subtitle file created: {srt_output_path.name}")

        except Exception as e:
            print(f"ERROR: Failed to process {video_file.name}. Error: {e}")

    print("\n--- Batch processing complete. ---")


if __name__ == "__main__":
    main()