from subtitlegen.asr.context import AsrContext
from subtitlegen.asr.whisper_context import (
    PROMPT_TOKEN_BUDGET,
    estimate_tokens,
    fit_whisper_context,
    is_decoder_overflow,
)


def test_fit_whisper_context_keeps_prompt_and_fits_hotwords() -> None:
    hotwords = tuple(f"Name{index:03d}" for index in range(200))
    fitted = fit_whisper_context(AsrContext("Canonical names: Luffy", hotwords))
    assert fitted is not None
    assert fitted.prompt == "Canonical names: Luffy"
    assert fitted.hotwords
    assert len(fitted.hotwords) < len(hotwords)
    used = estimate_tokens(fitted.prompt) + estimate_tokens(" ".join(fitted.hotwords))
    assert used <= PROMPT_TOKEN_BUDGET


def test_fit_whisper_context_drops_hotwords_when_prompt_fills_budget() -> None:
    prompt = "x" * (PROMPT_TOKEN_BUDGET * 4)
    fitted = fit_whisper_context(AsrContext(prompt, ("Luffy", "Zoro")))
    assert fitted is not None
    assert fitted.hotwords == ()
    assert estimate_tokens(fitted.prompt or "") <= PROMPT_TOKEN_BUDGET


def test_is_decoder_overflow_matches_whisper_error() -> None:
    error = RuntimeError(
        "No position encodings are defined for positions >= 448, but got position 449"
    )
    assert is_decoder_overflow(error)
    assert not is_decoder_overflow(RuntimeError("CUDA out of memory"))
