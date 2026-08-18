# Phase 1 Mac evidence

Device: MacBook Pro M4 Pro, 48 GB unified memory.

Backend: mlx-whisper 0.4.3, `mlx-community/whisper-large-v3-turbo`, English. Audio was decoded with PyAV; no host FFmpeg executable was installed.

## Results

- 12.01-second Avatar clip: 4.68 seconds, RTF 0.390.
- Full 5,931.05-second Avatar video: 350.47 seconds, RTF 0.059.
- Tuned full output: 979 cues, 2.22-second median, 6.00-second maximum, zero overlaps, zero cues over 8 seconds.
- Start, middle, and end cues parsed successfully from the generated SRT.
- A second output build from the persisted word artifact did not rerun ASR.

The full run exposed a recognition error (“isle” in place of a series term); Phase 2 glossary/context work owns that terminology correction rather than hiding it in timing logic.

## Packaging checks

- Installed `subtitlegen` and compatibility `python main.py` help commands passed.
- `docker compose config` passed with one reserved NVIDIA GPU and configurable video/model/job mounts.
- Automated tests cover paths containing spaces, recursive discovery, atomic files, corrupt manifests/artifacts, abandoned locks, concurrent manifest updates, resume, skip, overwrite, and forced benchmark behavior.
