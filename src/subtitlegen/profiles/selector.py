from __future__ import annotations

import math
from dataclasses import dataclass

from subtitlegen.asr.context import AsrContext
from subtitlegen.profiles.models import GlossaryEntry, SeriesProfile


@dataclass(frozen=True, slots=True)
class ContextBudget:
    max_prompt_tokens: int = 224
    max_hotwords: int = 64

    def __post_init__(self) -> None:
        if self.max_prompt_tokens <= 0 or self.max_hotwords <= 0:
            raise ValueError("context budgets must be positive")


class ContextSelector:
    def __init__(self, budget: ContextBudget | None = None) -> None:
        self._budget = budget or ContextBudget()

    def select(
        self,
        profile: SeriesProfile,
        *,
        arc: str | None = None,
        episode: str | None = None,
    ) -> AsrContext:
        entries = [
            entry for entry in profile.terms if entry.applies_to(arc=arc, episode=episode)
        ]
        entries.sort(key=lambda entry: self._priority(entry, arc=arc, episode=episode))
        prefix = f"Canonical {profile.title} spellings: "
        selected: list[str] = []
        for entry in entries:
            candidate = "; ".join((*selected, entry.canonical))
            if self._estimated_tokens(prefix + candidate) > self._budget.max_prompt_tokens:
                continue
            selected.append(entry.canonical)

        prompt = prefix + "; ".join(selected) if selected else None
        hotwords = tuple(entry.canonical for entry in entries[: self._budget.max_hotwords])
        return AsrContext(prompt=prompt, hotwords=hotwords)

    @staticmethod
    def _estimated_tokens(value: str) -> int:
        return math.ceil(len(value) / 4)

    @staticmethod
    def _priority(
        entry: GlossaryEntry,
        *,
        arc: str | None,
        episode: str | None,
    ) -> tuple[int, int, str]:
        scoped = bool(
            (arc is not None and entry.arcs)
            or (episode is not None and entry.episodes)
        )
        categories = {"character": 0, "place": 1, "term": 2}
        return (0 if scoped else 1, categories.get(entry.category, 3), entry.canonical.casefold())
