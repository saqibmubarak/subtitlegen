from __future__ import annotations

import numpy as np
import pytest

from subtitlegen.visual.furigana import mask_furigana


def test_mask_furigana_paints_over_small_ruby_above_the_line() -> None:
    pytest.importorskip("cv2")
    image = np.full((80, 160, 3), 255, dtype=np.uint8)
    image[40:70, 10:150] = 0
    image[8:18, 20:40] = 0
    image[8:18, 50:70] = 0

    masked = mask_furigana(image)

    assert masked[12, 30].mean() > 200
    assert masked[12, 60].mean() > 200
    assert masked[55, 80].mean() < 20
