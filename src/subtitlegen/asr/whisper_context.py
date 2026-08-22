from __future__ import annotations

import math

from subtitlegen.asr.context import AsrContext

# Whisper's decoder positions are 0..447. Prompt tokens and generated tokens share
# that window. Official prompting uses half for the prefix.
DECODER_POSITIONS = 448
PROMPT_TOKEN_BUDGET = 224


def estimate_tokens(value: str) -> int:
    return math.ceil(len(value) / 4) if value else 0


def fit_whisper_context(context: AsrContext | None) -> AsrContext | None:
    """Keep prompt plus hotwords inside Whisper's decoder prefix budget."""
    if context is None:
        return None
    remaining = PROMPT_TOKEN_BUDGET
    prompt = context.prompt
    if prompt:
        tokens = estimate_tokens(prompt)
        if tokens > remaining:
            prompt = _truncate_to_tokens(prompt, remaining)
            remaining = 0
        else:
            remaining -= tokens
    hotwords: list[str] = []
    for word in context.hotwords:
        cost = estimate_tokens(word) + (1 if hotwords else 0)
        if cost > remaining:
            continue
        hotwords.append(word)
        remaining -= cost
    if not prompt and not hotwords:
        return None
    return AsrContext(prompt=prompt, hotwords=tuple(hotwords))


def is_decoder_overflow(error: BaseException) -> bool:
    text = str(error).casefold()
    return "position encodings" in text or "positions >= 448" in text


def _truncate_to_tokens(value: str, token_budget: int) -> str | None:
    trimmed = value[: token_budget * 4].rsplit(" ", 1)[0].strip()
    return trimmed or None
