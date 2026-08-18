from typing import Any

import numpy as np
import pytest

from subtitlegen.errors import BackendOutOfMemoryError
from subtitlegen.profiles.models import GlossaryEntry, SeriesProfile
from subtitlegen.visual.models import OcrResult
from subtitlegen.visual.ocr import (
    MangaOcrEngine,
    contains_japanese,
    japanese_character_count,
)
from subtitlegen.visual.translation import NllbLocalTranslator


class FakeTokenizer:
    def __call__(self, text: str, **_kwargs: Any) -> dict[str, Any]:
        return {"source": text}

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "eng_Latn"
        return 42

    def batch_decode(self, _tokens: Any, **_kwargs: Any) -> list[str]:
        return ["Dofuramingo"]


class FakeTranslationModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.calls += 1
        assert kwargs["forced_bos_token_id"] == 42
        return [[1]]


def test_manga_ocr_engine_and_japanese_filter_contract() -> None:
    calls: list[Any] = []

    def model(image: Any) -> str:
        calls.append(image)
        return " 日本 "

    engine = MangaOcrEngine(
        model_factory=lambda: model,
        image_factory=lambda image: image,
    )
    result = engine.recognize(np.zeros((5, 5, 3), dtype=np.uint8))
    assert result == OcrResult("日本")
    assert contains_japanese(result.text)
    assert japanese_character_count("日本, English") == 2
    assert not contains_japanese("English only")
    engine.close()
    engine.recognize(np.zeros((5, 5, 3), dtype=np.uint8))
    assert len(calls) == 2


def test_nllb_translator_caches_and_applies_profile_canonicalization() -> None:
    model = FakeTranslationModel()
    loads: list[str] = []
    profile = SeriesProfile(
        schema_version=1,
        profile_id="one-piece",
        title="One Piece",
        language="en",
        terms=(GlossaryEntry("Doflamingo", aliases=("Dofuramingo",)),),
        visual_translations=(("一人はぐれた錦えもん", "Kin'emon Gets Separated"),),
    )

    def factory(model_name: str, device: str) -> tuple[FakeTokenizer, FakeTranslationModel]:
        loads.append(f"{model_name}:{device}")
        return FakeTokenizer(), model

    translator = NllbLocalTranslator(profile=profile, model_factory=factory)
    assert translator.translate("一人はぐれた錦えもん") == "Kin'emon Gets Separated"
    assert translator.translate("一人はぐれた錦えも") == "Kin'emon Gets Separated"
    assert model.calls == 0
    assert translator.translate("ドフラミンゴ") == "Doflamingo"
    assert translator.translate("ドフラミンゴ") == "Doflamingo"
    assert model.calls == 1
    assert len(loads) == 1
    translator.close()
    assert translator.translate("ドフラミンゴ") == "Doflamingo"
    assert len(loads) == 2
    with pytest.raises(ValueError):
        translator.translate(" ")


def test_nllb_translator_provides_oom_guidance() -> None:
    class OomModel(FakeTranslationModel):
        def generate(self, **_kwargs: Any) -> Any:
            raise RuntimeError("out of memory")

    translator = NllbLocalTranslator(
        model_factory=lambda *_args: (FakeTokenizer(), OomModel())
    )
    with pytest.raises(BackendOutOfMemoryError, match="releasing ASR"):
        translator.translate("日本")
