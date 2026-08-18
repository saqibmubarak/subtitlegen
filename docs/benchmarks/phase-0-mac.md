# Phase 0 Mac evidence

Device: MacBook Pro M4 Pro, 48 GB unified memory.

Sample: 12-second PyAV-extracted speech interval at 00:01:54 from the local Avatar video.

Backend: cached faster-whisper large-v3-turbo, CPU int8, English, beam size 1.

- Wall time: 6.60 seconds
- Words: 28
- Cues: 3
- Median cue duration: 3.88 seconds
- Maximum cue duration: 4.02 seconds
- Cues over 8 seconds: 0
- Overlaps: 0

The smoke output mentions “airbender” and “avatar” and tracks three readable speech phrases. The ignored full-video baseline contains 282 cues, more than 100 cues over 8 seconds, and a maximum cue longer than 400 seconds. Full 99-minute regeneration remains assigned to MLX in Phase 1 and RTX acceptance in Phase 5.
