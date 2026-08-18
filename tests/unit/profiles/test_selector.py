import pytest

from subtitlegen.asr.context import AsrContext
from subtitlegen.profiles.models import GlossaryEntry, SeriesProfile
from subtitlegen.profiles.selector import ContextBudget, ContextSelector


def _profile() -> SeriesProfile:
    return SeriesProfile(
        1,
        "one-piece",
        "One Piece",
        "en",
        (
            GlossaryEntry("Luffy", category="character"),
            GlossaryEntry("Dressrosa", category="place", arcs=("Dressrosa",)),
            GlossaryEntry("Wano", category="place", arcs=("Wano",)),
            GlossaryEntry("Haki", category="term"),
        ),
    )


def test_asr_context_and_budget_validate() -> None:
    assert AsrContext(prompt="Aang", hotwords=("Aang",)).prompt == "Aang"
    with pytest.raises(ValueError):
        AsrContext(prompt=" ")
    with pytest.raises(ValueError):
        ContextBudget(max_prompt_tokens=0)


def test_selector_prioritizes_scope_and_honors_budgets() -> None:
    selector = ContextSelector(ContextBudget(max_prompt_tokens=16, max_hotwords=2))
    context = selector.select(_profile(), arc="Dressrosa")
    assert context.hotwords[0] == "Dressrosa"
    assert "Wano" not in context.hotwords
    assert len(context.hotwords) == 2
    assert context.prompt is not None and len(context.prompt) <= 64


def test_selector_returns_no_prompt_when_prefix_exhausts_budget() -> None:
    context = ContextSelector(ContextBudget(max_prompt_tokens=1)).select(_profile())
    assert context.prompt is None
