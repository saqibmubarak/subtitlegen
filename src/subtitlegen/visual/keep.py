from __future__ import annotations

import re
from collections.abc import Iterable

from subtitlegen.evaluation.metrics import character_error_rate
from subtitlegen.visual.models import VisualEvent
from subtitlegen.visual.ocr import (
    has_title_script,
    hiragana_character_count,
    kanji_character_count,
    katakana_character_count,
)

_DIALOGUE_ENGLISH = re.compile(
    r"\b(i'm|i am|let's|don't|going to|not going|a little bit|"
    r"is a|the following|i'm not sure|not a |not going to lie)\b",
    re.IGNORECASE,
)
_DATE_TEXT = re.compile(r"[0-9０-９]{2,4}\s*年")
_HUD_LOANWORDS = re.compile(
    r"インターネット|インストール|メール|サービス|ノート|ネット"
)
_SIMPLIFIED_ONLY = re.compile(r"[宾广兴鱼乐门车队钟宝闲]")
_REPEATED_CHAR = re.compile(r"^(.)\1{2,}$")


def keep_visual_events(
    events: Iterable[VisualEvent],
    *,
    glossary: Iterable[str] = (),
) -> tuple[VisualEvent, ...]:
    names = {term.casefold() for term in glossary if term}
    kept = tuple(event for event in events if should_keep_event(event, glossary=names))
    return collapse_overlapping_events(kept)


def should_keep_event(
    event: VisualEvent,
    *,
    glossary: Iterable[str] = (),
) -> bool:
    source = event.source_text.strip()
    if not source or not has_title_script(source):
        return False
    if _DATE_TEXT.search(source) or _HUD_LOANWORDS.search(source):
        return False
    compact = re.sub(r"\s+", "", source)
    if _REPEATED_CHAR.match(compact):
        return False
    hiragana = hiragana_character_count(source)
    katakana = katakana_character_count(source)
    kanji = kanji_character_count(source)
    content = kanji + katakana
    if hiragana > content * 2 and content < 3:
        return False
    names = {term.casefold() for term in glossary if term}
    if _glossary_hit(event, names):
        return True
    if katakana == 0 and kanji < 4:
        return False
    if _SIMPLIFIED_ONLY.search(source) and katakana == 0:
        return False
    translated = event.translated_text.strip()
    if translated and _DIALOGUE_ENGLISH.search(translated):
        if content < 6 or event.box.y < 400:
            return False
    return True


def collapse_overlapping_events(
    events: Iterable[VisualEvent],
) -> tuple[VisualEvent, ...]:
    remaining = sorted(events, key=lambda event: (event.start, event.end))
    chosen: list[VisualEvent] = []
    while remaining:
        current = remaining.pop(0)
        group = [current]
        keep: list[VisualEvent] = []
        for other in remaining:
            if _same_card(current, other):
                group.append(other)
            else:
                keep.append(other)
        chosen.append(_best_event(group))
        remaining = keep
    return tuple(chosen)


def _same_card(left: VisualEvent, right: VisualEvent) -> bool:
    if left.end < right.start - 1.5 or right.end < left.start - 1.5:
        return False
    a = left.source_text.strip()
    b = right.source_text.strip()
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return character_error_rate(a, b) <= 0.4


def _best_event(group: list[VisualEvent]) -> VisualEvent:
    return max(
        group,
        key=lambda event: (
            kanji_character_count(event.source_text)
            + katakana_character_count(event.source_text),
            len(event.source_text),
            -event.start,
            event.end - event.start,
        ),
    )


def _glossary_hit(event: VisualEvent, names: set[str]) -> bool:
    if not names:
        return False
    haystacks = (event.source_text.casefold(), event.translated_text.casefold())
    return any(name in haystack for haystack in haystacks for name in names)
