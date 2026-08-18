# 2. Sync Fix

## Evidence

The current Avatar output contains 282 cues:

- median duration: 8.3 seconds;
- 99 cues exceed 10 seconds;
- 12 cues exceed 60 seconds;
- the worst cue spans 464 seconds for about 100 characters.

This is a segmentation failure, not a minor timestamp offset.

## Cause

The current faster-whisper call enables VAD but keeps its defaults and disables word timestamps. Removed silence is later mapped back to the source timeline, allowing a short segment to span a long gap. Two GPU workers also load two model copies into 8 GB VRAM.

## Change

Keep faster-whisper and:

```text
word_timestamps = true
condition_on_previous_text = false
hallucination_silence_threshold = 2.0
vad.min_silence_duration_ms = 500
vad.speech_pad_ms = 200
vad.max_speech_duration_s = 30
```

Build cues from words, flushing on:

- duration above 6 seconds;
- text above about 84 characters;
- silence above 0.9 seconds;
- sentence-ending punctuation when the cue is readable.

Use `float16`, one GPU worker, and in-process batching on the RTX 3070 Ti. These values must remain configurable and be validated against the sample rather than treated as universal constants.

## Trade-off

| Choice | Speed | Timing | Decision |
|---|---:|---:|---|
| Current turbo/int8 segments | Highest | Unacceptable | Remove |
| Turbo/fp16 word cues | High | Good | Fast preset |
| Large-v3/fp16 word cues | Lower | Better | Quality preset |
| WhisperX alignment | Additional pass | Best Whisper timing | Evaluate in Phase 3 |

Official faster-whisper benchmarks show large Whisper inference fitting an RTX 3070 Ti in fp16; exact speed and memory depend on model, batch size, and audio [R1]. WhisperX documents forced alignment because native Whisper segment timestamps can differ by seconds [R2].

## Validation

Regenerate the same Avatar file and require:

- no cue above 8 seconds unless explicitly allowed;
- median duration between 2 and 4 seconds;
- no cue spanning a long music/action gap;
- monotonically increasing, non-overlapping timestamps;
- manual checks at episode start, middle, and end.

Next: [Upgrade choices](03-upgrade-choices.md). Sources: [References](05-references.md).
