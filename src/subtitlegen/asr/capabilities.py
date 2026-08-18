from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    word_timestamps: bool
    context_prompt: bool
    hotwords: bool
    requires_cuda: bool
