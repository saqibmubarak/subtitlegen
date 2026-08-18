from subtitlegen.domain.models import Cue
from subtitlegen.visual.models import StyledCue, VisualEvent


class SubtitleMerger:
    def merge(
        self,
        dialogue: list[Cue],
        visual: list[VisualEvent],
    ) -> tuple[StyledCue, ...]:
        events = [
            StyledCue(cue.start, cue.end, cue.text, "Dialogue")
            for cue in dialogue
        ]
        events.extend(
            StyledCue(
                event.start,
                event.end,
                event.translated_text,
                event.category,
            )
            for event in visual
        )
        return tuple(
            sorted(
                events,
                key=lambda event: (
                    event.start,
                    0 if event.style == "Dialogue" else 1,
                    event.end,
                ),
            )
        )
