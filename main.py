# main.py

import argparse
import sys
from pathlib import Path
import torch
import multiprocessing
import os

# Import project modules
from config import load_config
from transcriber import transcribe_video



def format_timestamp(seconds: float) -> str:
    if seconds is None: return "00:00:00,000"
    milliseconds = int(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    seconds = milliseconds // 1_000
    milliseconds %= 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(segments, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
            f.write(f"{segment['text'].strip()}\n\n")


def find_video_files(input_path: Path, extensions):
    video_files = []
    if input_path.is_file():
        if input_path.suffix.lower() in extensions: video_files.append(input_path)
    elif input_path.is_dir():
        for ext in extensions: video_files.extend(input_path.rglob(f'*{ext}'))
    return [p.resolve() for p in video_files]


# --- End Utility Placeholder ---


# --- WORKER FUNCTION FOR MULTIPROCESSING ---
def process_single_file(file_data_and_config):
    """
    Worker function executed by the multiprocessing pool.
    Handles loading the model (once per worker), transcription, and saving the SRT.
    """
    video_file, config = file_data_and_config

    model_identifier = config['model_identifier']
    device = config['device']
    language = config['language']
    compute_type = config['compute_type']  # Passed to transcriber

    print(f"[Worker {os.getpid()}] Processing: {video_file.name}")

    try:
        # 1. Transcribe (model is efficiently cached inside transcriber.py)
        result = transcribe_video(
            video_file,
            model_identifier,
            device,
            language,
            compute_type
        )

        # 2. Define Output Path (Same path, same basename, .srt extension)
        srt_output_path = video_file.with_suffix('.srt')

        # 3. Write SRT
        write_srt(result['segments'], srt_output_path)

        print(f"[Worker {os.getpid()}] SUCCESS: Subtitle file created: {srt_output_path.name}")
        return True
    except Exception as e:
        print(f"[Worker {os.getpid()}] ERROR: Failed to process {video_file.name}. Error: {e}")
        return False


def main():
    """Main execution function for the CLI application."""

    parser = argparse.ArgumentParser(
        description="A command-line tool to generate SRT subtitles for video files using Faster Whisper.",
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

    # --- 1. Load Configuration & Setup ---
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Fatal Error: Failed to load configuration. {e}")
        sys.exit(1)

    input_path = Path(args.input_path).resolve()
    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)

    # Check GPU availability
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA device not found. Transcription will use CPU.")
        device = 'cpu'
    config['device'] = device

    # --- 2. Find Files ---
    print(f"--- Configuration ---")
    print(
        f"Model: {config['model_identifier']}, Device: {config['device']}, Compute: {config['compute_type']}, Language: {config['language'] or 'Auto'}")

    video_files = find_video_files(input_path, config['video_extensions'])

    if not video_files:
        print(f"No supported video files found in {input_path} (or its subdirectories).")
        return

    max_workers = min(config['parallel_workers'], len(video_files)) if video_files else 1
    print(f"Found {len(video_files)} file(s) for processing. Using {max_workers} worker(s) in parallel.")

    # --- 3. Process Files (Parallelized) ---

    tasks = [(video_file, config) for video_file in video_files]

    if max_workers > 1 and config['device'] == 'cuda':
        # Use a Pool to manage worker processes for parallel GPU transcription
        try:
            with multiprocessing.Pool(processes=max_workers) as pool:
                pool.map(process_single_file, tasks)
        except Exception as e:
            print(f"FATAL ERROR during multiprocessing. Error: {e}")
            print("Falling back to sequential processing...")
            max_workers = 1  # Fallback

    if max_workers == 1 or config['device'] == 'cpu':
        # Sequential processing (fallback or intentional CPU mode)
        print("Running sequentially.")
        for file_data_and_config in tasks:
            process_single_file(file_data_and_config)

    print("\n--- Batch processing complete. ---")


if __name__ == "__main__":
    main()