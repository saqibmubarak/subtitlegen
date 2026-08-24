"""Paddle detect/rec child process. Keep PyTorch out of this module."""

from __future__ import annotations

from typing import Any

import numpy as np


def construct_engine(factory: Any, options: dict[str, Any]) -> Any:
    try:
        return factory(**options)
    except TypeError:
        options = dict(options)
        options.pop("engine_config", None)
        options.pop("engine", None)
        try:
            return factory(**options)
        except TypeError:
            options.pop("enable_mkldnn", None)
            return factory(**options)


def load_engines(device_type: str) -> tuple[Any, Any]:
    from paddleocr import TextDetection, TextRecognition

    from subtitlegen.visual.detection import (
        disable_paddle_onednn,
        text_detection_options,
        text_recognition_options,
    )

    disable_paddle_onednn()
    detector = construct_engine(TextDetection, text_detection_options(device_type=device_type))
    recognizer = construct_engine(
        TextRecognition,
        text_recognition_options(device_type=device_type),
    )
    return detector, recognizer


def run_worker(request_queue: Any, response_queue: Any, device_type: str) -> None:
    from subtitlegen.visual.detection import PaddleOcrDetector
    from subtitlegen.visual.ocr import PaddleTextRecognizer

    detector_engine, recognizer_engine = load_engines(device_type)
    detector = PaddleOcrDetector(engine_factory=lambda: detector_engine)
    recognizer = PaddleTextRecognizer(engine_factory=lambda: recognizer_engine)
    while True:
        message = request_queue.get()
        operation = message.get("op")
        if operation == "stop":
            response_queue.put({"ok": True})
            return
        if operation == "ping":
            response_queue.put({"ok": True, "device": device_type})
            continue
        try:
            if operation == "detect":
                boxes = detector.detect(np.asarray(message["image"]))
                response_queue.put({"ok": True, "boxes": _boxes_payload(boxes)})
            elif operation == "detect_batch":
                batches = detector.detect_batch(
                    [np.asarray(image) for image in message["images"]]
                )
                response_queue.put(
                    {
                        "ok": True,
                        "batches": [_boxes_payload(boxes) for boxes in batches],
                    }
                )
            elif operation == "recognize":
                text = recognizer.recognize(np.asarray(message["image"])).text
                response_queue.put({"ok": True, "text": text})
            else:
                response_queue.put({"ok": False, "error": f"unknown op {operation}"})
        except Exception as error:
            response_queue.put({"ok": False, "error": str(error)})


def _boxes_payload(boxes: Any) -> list[dict[str, float | int]]:
    return [
        {
            "x": box.x,
            "y": box.y,
            "width": box.width,
            "height": box.height,
            "score": box.score,
        }
        for box in boxes
    ]
