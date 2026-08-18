# 🎥 Subtitler CLI: Batch ASR Transcription Service (Whisper-Powered)

This is a **Python command-line application** designed for efficient, batch-processed Automated Speech Recognition (ASR) transcription of video files using the **OpenAI Whisper model**. It is engineered for simplicity, speed (via GPU support), and robust configuration.

## ✨ Features

* **Command Line Interface (CLI):** Entirely console-based, no GUI required.
* **Batch Processing:** Recursively traverses an input directory and processes all supported video files in it and its subdirectories.
* **Configurable Models:** Easily switch between Whisper models (`tiny`, `base`, `small`) and add new ones via the `config.ini` file.
* **GPU Acceleration:** Configurable setting (`device = cuda`) to leverage CUDA-enabled GPUs for faster processing.
* **Standard Output Format & Placement:** Generates standard **`.srt`** subtitle files, placed in the **same directory** as the source video file and using the same base filename.

---

## 🏗️ Project Structure

The project is structured into modular Python files:

| File Name | Description |
| :--- | :--- |
| `main.py` | Main entry point; handles CLI arguments, file discovery, and orchestrates processing. |
| `config.py` | Utility functions for loading and parsing settings from `config.ini`. |
| `transcriber.py` | Core logic for Whisper model loading (with caching) and audio transcription. |
| `utils.py` | Utilities for video file discovery (`os.walk` equivalent) and SRT format generation. |
| `config.ini` | Configuration file for models, device settings, and file extensions. |
| `requirements.txt`| List of required Python packages (`torch`, `openai-whisper`, `ffmpeg-python`). |

---

## ⚙️ Prerequisites

To run this tool, you need the following:

1.  **Python 3.8+**
2.  **FFmpeg:** This is mandatory for the Whisper library to handle video and audio extraction. You must install `ffmpeg` on your system (e.g., via `brew`, `apt`, or by downloading the binaries).

---

## 🚀 Installation

1.  **Clone or Download** the project files.

2.  **Install Python Dependencies:** The core application relies on `torch` (for GPU support) and `openai-whisper`.

    ```bash
    pip install -r requirements.txt
    ```

    *If you intend to use a GPU (`device = cuda`), ensure your PyTorch installation is compatible with your CUDA drivers.*

---

## 📝 Configuration (`config.ini`)

The core behavior is controlled by `config.ini`.

```ini
[MODELS]
# Define available Whisper models. New models can be added by their identifier.
tiny = tiny
base = base
small = small

[TRANSCRIPTION]
# Device to use for running the model.
# Set to 'cuda' to use your NVIDIA GPU (recommended for speed).
# Set to 'cpu' if no GPU is available or preferred.
device = cuda

# Which model to use from the [MODELS] section (e.g., 'base').
model_name = base

# Language of the video, or 'None' for automatic detection (e.g., 'en', 'es').
language = None

[FILES]
# Supported video file extensions (used for recursive searching).
video_extensions = .mp4, .mkv, .avi, .mov, .wmv