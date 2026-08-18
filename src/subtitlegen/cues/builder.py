from __future__ import annotations

from collections.abc import Iterable

from subtitlegen.cues.rules import CueRules
from subtitlegen.domain.models import Cue, Word

_SENTENCE_ENDINGS = (".", "!", "?", "\u3002", "\uff01", "\uff1f")


class CueBuilder:
    """Build readable, bounded cues from timestamped words."""

    def __init__(self, rules: CueRules | None = None) -> None:
        self._rules = rules or CueRules()

    def build(self, words: Iterable[Word]) -> list[Cue]:
        cues: list[Cue] = []
        buffer: list[Word] = []

        for word in words:
            if word.end - word.start > self._rules.max_duration_seconds:
                if buffer:
                    cues.append(self._to_cue(buffer))
                    buffer = []
                continue
            if buffer and self._must_flush_before(buffer, word):
                cues.append(self._to_cue(buffer))
                buffer = []

            buffer.append(word)
            if self._should_flush_after(buffer):
                cues.append(self._to_cue(buffer))
                buffer = []

        if buffer:
            cues.append(self._to_cue(buffer))

        return self._remove_overlaps(cues)

    def _must_flush_before(self, words: list[Word], next_word: Word) -> bool:
        gap = max(0.0, next_word.start - words[-1].end)
        duration = next_word.end - words[0].start
        proposed_text = self._join([*words, next_word])
        return (
            gap > self._rules.max_gap_seconds
            or duration > self._rules.max_duration_seconds
            or len(proposed_text) > self._rules.max_characters
        )

    def _should_flush_after(self, words: list[Word]) -> bool:
        text = self._join(words)
        duration = words[-1].end - words[0].start
        return (
            text.endswith(_SENTENCE_ENDINGS)
            and duration >= self._rules.punctuation_flush_min_seconds
        )

    @staticmethod
    def _join(words: Iterable[Word]) -> str:
        pieces = [word.text for word in words]
        if any(piece[:1].isspace() for piece in pieces):
            return "".join(pieces).strip()
        return " ".join(piece.strip() for piece in pieces).strip()

    def _to_cue(self, words: list[Word]) -> Cue:
        start = words[0].start
        return Cue(start=start, end=words[-1].end, text=self._join(words))

    @staticmethod
    def _remove_overlaps(cues: list[Cue]) -> list[Cue]:
        normalized: list[Cue] = []
        for cue in cues:
            start = max(cue.start, normalized[-1].end if normalized else 0.0)
            end = max(start, cue.end)
            normalized.append(Cue(start=start, end=end, text=cue.text))
        return normalized
