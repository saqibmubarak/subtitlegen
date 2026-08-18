from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AsrContext:
    prompt: str | None = None
    hotwords: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("ASR prompt must not be blank")
        if any(not term.strip() for term in self.hotwords):
            raise ValueError("ASR hotwords must not be blank")
