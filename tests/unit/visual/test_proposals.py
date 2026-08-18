from typing import Any

import numpy as np
import pytest

from subtitlegen.visual.models import BoundingBox
from subtitlegen.visual.proposals import TemporalDifferenceProposer


def _component_extractor(
    _mask: np.ndarray[Any, Any],
) -> tuple[tuple[int, int, int, int, int], ...]:
    return ((8, 9, 8, 5, 40),)


def _changed_component_extractor(
    mask: np.ndarray[Any, Any],
) -> tuple[tuple[int, int, int, int, int], ...]:
    return _component_extractor(mask) if mask.any() else ()


def test_temporal_proposer_uses_full_scene_then_changed_regions_and_hold() -> None:
    proposer = TemporalDifferenceProposer(
        analysis_width=32,
        hold_frames=2,
        padding_ratio=0,
        component_extractor=_component_extractor,
    )
    black = np.zeros((100, 200, 3), dtype=np.uint8)
    changed_image = black.copy()
    changed_image[40:80, 40:120] = 255

    assert proposer.propose(black) == (BoundingBox(0, 0, 200, 100),)
    changed = proposer.propose(changed_image)
    assert BoundingBox(50, 50, 50, 28) in changed
    assert proposer.propose(changed_image)
    assert proposer.propose(changed_image, scene_change=True) == changed
    proposer.reset()
    assert proposer.propose(black) == (BoundingBox(0, 0, 200, 100),)


def test_temporal_proposer_validates_configuration_and_images() -> None:
    with pytest.raises(ValueError):
        TemporalDifferenceProposer(difference_threshold=0)
    with pytest.raises(ValueError):
        TemporalDifferenceProposer(
            minimum_changed_area_ratio=0.5,
            maximum_changed_area_ratio=0.4,
        )
    with pytest.raises(ValueError):
        TemporalDifferenceProposer(hold_frames=0)
    proposer = TemporalDifferenceProposer(component_extractor=_component_extractor)
    with pytest.raises(ValueError):
        proposer.propose(np.array([]))


def test_temporal_proposer_keeps_static_card_region_for_timing() -> None:
    proposer = TemporalDifferenceProposer(
        analysis_width=32,
        hold_frames=6,
        full_frame_hold_frames=1,
        padding_ratio=0,
        component_extractor=_changed_component_extractor,
    )
    background = np.zeros((100, 200, 3), dtype=np.uint8)
    card = background.copy()
    card[40:80, 40:120] = 255
    expected = (BoundingBox(50, 50, 50, 28),)

    assert proposer.propose(background) == (BoundingBox(0, 0, 200, 100),)
    assert proposer.propose(card) == expected
    for _ in range(5):
        assert proposer.propose(card) == expected
    assert proposer.propose(card) == ()


def test_temporal_proposer_opencv_components_find_changed_region() -> None:
    pytest.importorskip("cv2")
    proposer = TemporalDifferenceProposer(
        analysis_width=64,
        hold_frames=2,
        full_frame_hold_frames=1,
    )
    background = np.zeros((100, 200, 3), dtype=np.uint8)
    changed = background.copy()
    changed[60:80, 80:140] = 255

    proposer.propose(background)
    regions = proposer.propose(changed)

    assert regions
    assert any(region.y + region.height / 2 >= 50 for region in regions)
