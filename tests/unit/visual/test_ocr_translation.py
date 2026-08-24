from typing import Any

import numpy as np
import pytest

from subtitlegen.errors import BackendOutOfMemoryError
from subtitlegen.profiles.models import GlossaryEntry, SeriesProfile
from subtitlegen.visual.models import OcrResult
from subtitlegen.visual.ocr import (
    MangaOcrEngine,
    PaddleTextRecognizer,
    contains_japanese,
    has_title_script,
    hiragana_character_count,
    japanese_character_count,
    rotate_vertical_crop,
    warmup_torch,
)
from subtitlegen.visual.translation import NllbLocalTranslator


class FakeTokenizer:
    def __call__(self, text: str | list[str], **_kwargs: Any) -> dict[str, Any]:
        return {"source": text}

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "eng_Latn"
        return 42

    def batch_decode(self, tokens: Any, **_kwargs: Any) -> list[str]:
        count = len(tokens) if isinstance(tokens, list) else 1
        return ["Dofuramingo"] * count


class FakeTranslationModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.calls += 1
        assert kwargs["forced_bos_token_id"] == 42
        source = kwargs.get("source")
        count = len(source) if isinstance(source, list) else 1
        return [[1]] * count


def test_manga_ocr_engine_and_japanese_filter_contract() -> None:
    calls: list[Any] = []

    def model(image: Any) -> str:
        calls.append(image)
        return " 日本 "

    engine = MangaOcrEngine(
        model_factory=lambda: model,
        image_factory=lambda image: image,
    )
    engine.warmup()
    result = engine.recognize(np.zeros((5, 5, 3), dtype=np.uint8))
    assert result == OcrResult("日本")
    assert contains_japanese(result.text)
    assert japanese_character_count("日本, English") == 2
    assert hiragana_character_count("そういえば") == 5
    assert not contains_japanese("English only")
    assert has_title_script("ドレスローザ")
    assert has_title_script("立場破壊を侍救出チーム")
    assert has_title_script("一人はぐれた錦えもん")
    assert not has_title_script("そういえば、")
    assert not has_title_script("人のところで、")
    tall = np.zeros((20, 6), dtype=np.uint8)
    assert rotate_vertical_crop(tall).shape[:2] == (6, 20)
    assert rotate_vertical_crop(np.zeros((6, 20), dtype=np.uint8)).shape[:2] == (6, 20)
    engine.close()
    engine.recognize(np.zeros((5, 5, 3), dtype=np.uint8))
    assert len(calls) == 2


def test_paddle_recognizer_reads_rec_text_payload() -> None:
    class FakeEngine:
        def predict(self, _image: Any) -> list[dict[str, str]]:
            return [{"rec_text": " ドレスローザ "}]

    recognizer = PaddleTextRecognizer(engine_factory=FakeEngine)
    assert recognizer.recognize(np.zeros((8, 8), dtype=np.uint8)).text == "ドレスローザ"
    recognizer.close()


def test_paddle_recognizer_ignores_image_payload_dicts() -> None:
    class FakeEngine:
        def predict(self, _image: Any) -> list[dict[str, Any]]:
            return [{"input_path": None, "input_img": np.zeros((2, 2), dtype=np.uint8)}]

    recognizer = PaddleTextRecognizer(engine_factory=FakeEngine)
    assert recognizer.recognize(np.zeros((8, 8), dtype=np.uint8)).text == ""
    recognizer.close()


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


def test_nllb_translate_many_batches_unique_uncached_strings() -> None:
    model = FakeTranslationModel()
    translator = NllbLocalTranslator(
        model_factory=lambda *_args: (FakeTokenizer(), model)
    )

    assert translator.translate_many(["ドフラミンゴ", "日本", "ドフラミンゴ"]) == [
        "Dofuramingo",
        "Dofuramingo",
        "Dofuramingo",
    ]
    assert model.calls == 1
    assert translator.translate_many(["日本"]) == ["Dofuramingo"]
    assert model.calls == 1
    assert translator.translate_many([]) == []


def test_warmup_torch_allocates_one_tensor() -> None:
    warmup_torch()
