from typing import Any

import numpy as np
import pytest

from subtitlegen.visual.models import BoundingBox, OcrResult
from subtitlegen.visual.presence import JapaneseCharacterScanner


class FakeDetector:
    def __init__(self, boxes: tuple[BoundingBox, ...]) -> None:
        self.boxes = boxes

    def detect(self, _image: Any) -> tuple[BoundingBox, ...]:
        return self.boxes


class FakeRecognizer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.closed = False

    def recognize(self, _image: Any) -> OcrResult:
        return OcrResult(self.text)

    def close(self) -> None:
        self.closed = True


def test_japanese_character_scanner_accepts_any_japanese_crop() -> None:
    scanner = JapaneseCharacterScanner(
        FakeDetector((BoundingBox(0, 0, 4, 4),)),
        FakeRecognizer("ドレスローザ"),
        analysis_width=32,
    )
    assert scanner.contains_japanese(np.zeros((64, 64, 3), dtype=np.uint8))
    scanner.close()


def test_japanese_character_scanner_rejects_non_japanese_and_empty_boxes() -> None:
    empty = JapaneseCharacterScanner(
        FakeDetector(()),
        FakeRecognizer("ドレスローザ"),
    )
    english = JapaneseCharacterScanner(
        FakeDetector((BoundingBox(0, 0, 4, 4),)),
        FakeRecognizer("Dressrosa"),
    )
    empty_decision = empty.inspect(np.zeros((16, 16), dtype=np.uint8))
    english_decision = english.inspect(np.zeros((16, 16), dtype=np.uint8))
    assert not empty_decision.accepted
    assert empty_decision.reason == "no_boxes"
    assert not english_decision.accepted
    assert english_decision.reason == "no_japanese"
    assert english_decision.recognized == ("Dressrosa",)
    assert not empty.contains_japanese(np.zeros((16, 16), dtype=np.uint8))
    assert not english.contains_japanese(np.zeros((16, 16), dtype=np.uint8))
    with pytest.raises(ValueError):
        JapaneseCharacterScanner(FakeDetector(()), FakeRecognizer("日"), analysis_width=8)


def test_japanese_character_scanner_finds_japanese_behind_larger_english_boxes() -> None:
    class MixedRecognizer:
        def recognize(self, image: Any) -> OcrResult:
            width = int(np.asarray(image).shape[1])
            return OcrResult("Scene-4" if width >= 20 else "ドレスローザ")

    large = BoundingBox(0, 0, 30, 10)
    small = BoundingBox(0, 12, 8, 8)
    hidden = JapaneseCharacterScanner(
        FakeDetector((large, small)),
        MixedRecognizer(),
        analysis_width=64,
        maximum_crops=2,
    )
    capped = JapaneseCharacterScanner(
        FakeDetector((large, small)),
        MixedRecognizer(),
        analysis_width=64,
        maximum_crops=1,
    )
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    decision = hidden.inspect(image)
    assert decision.accepted
    assert decision.reason == "hit"
    assert "ドレスローザ" in decision.recognized
    assert decision.boxes
    assert not capped.contains_japanese(image)


def test_japanese_character_scanner_rotates_vertical_crops_for_horizontal_ocr() -> None:
    class HorizontalOnlyRecognizer:
        def recognize(self, image: Any) -> OcrResult:
            height, width = np.asarray(image).shape[:2]
            return OcrResult("ドレスローザ" if width > height else "Scene-4")

    scanner = JapaneseCharacterScanner(
        FakeDetector((BoundingBox(0, 0, 6, 20),)),
        HorizontalOnlyRecognizer(),
        analysis_width=32,
    )
    decision = scanner.inspect(np.zeros((32, 32, 3), dtype=np.uint8))
    assert decision.accepted
    assert decision.recognized == ("ドレスローザ",)
    assert decision.orientations == ("vertical-rotated",)
    assert decision.boxes == (BoundingBox(0, 0, 6, 20),)


def test_japanese_character_scanner_rejects_hiragana_filler() -> None:
    scanner = JapaneseCharacterScanner(
        FakeDetector((BoundingBox(0, 0, 4, 4),)),
        FakeRecognizer("そういえば"),
        analysis_width=32,
    )
    decision = scanner.inspect(np.zeros((16, 16), dtype=np.uint8))
    assert not decision.accepted
    assert decision.reason == "weak_japanese"
    assert decision.boxes == ()
