from __future__ import annotations

from typing import Any

import numpy as np


def mask_furigana(image: Any, *, vertical: bool = False) -> Any:
    """Paint over small ruby components so OCR reads the main line only.

    Furigana sits above a horizontal kanji line (or beside a vertical column)
    and is much smaller than the body text. Mixing both sizes into one crop
    is what turns ``工場破壊＆侍救出チーム`` into a fluent wrong sentence.
    """
    array = np.asarray(image)
    if array.ndim < 2 or min(array.shape[:2]) < 16:
        return array
    binary = _text_mask(array)
    if binary is None or int(binary.sum()) < 8:
        return array
    components = _components(binary)
    if components is None:
        return array
    num, labels, stats, centroids = components
    if num <= 2:
        return array
    heights = stats[1:, 3].astype(np.float64)
    widths = stats[1:, 2].astype(np.float64)
    main_h = float(np.percentile(heights, 75))
    main_w = float(np.percentile(widths, 75))
    if main_h < 8 or main_w < 8:
        return array
    main_rows = heights >= 0.7 * main_h
    main_cols = widths >= 0.7 * main_w
    if not main_rows.any() or not main_cols.any():
        return array
    median_cy = float(np.median(centroids[1:][main_rows, 1]))
    median_cx = float(np.median(centroids[1:][main_cols, 0]))
    background = _background(array, binary)
    output = array.copy()
    masked = False
    for index in range(1, num):
        height = float(stats[index, 3])
        width = float(stats[index, 2])
        cx, cy = centroids[index]
        if vertical:
            ruby = width < 0.55 * main_w and cx > median_cx
        else:
            ruby = height < 0.55 * main_h and cy < median_cy
        if not ruby:
            continue
        output[labels == index] = background
        masked = True
    return output if masked else array


def _text_mask(array: np.ndarray[Any, Any]) -> np.ndarray[Any, Any] | None:
    gray = array.mean(axis=2) if array.ndim == 3 else array.astype(np.float64)
    threshold = float(np.mean(gray))
    dark = gray < threshold
    if float(dark.mean()) > 0.55:
        dark = gray > threshold
    return (dark.astype(np.uint8) * 255) if dark.any() else None


def _components(
    binary: np.ndarray[Any, Any],
) -> tuple[int, np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
    try:
        import cv2
    except ImportError:
        return None
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    return int(count), labels, stats, centroids


def _background(array: np.ndarray[Any, Any], text_mask: np.ndarray[Any, Any]) -> Any:
    paper = text_mask == 0
    if array.ndim == 3:
        if not paper.any():
            return (255, 255, 255)
        return tuple(int(value) for value in np.median(array[paper], axis=0))
    if not paper.any():
        return 255
    return int(np.median(array[paper]))
