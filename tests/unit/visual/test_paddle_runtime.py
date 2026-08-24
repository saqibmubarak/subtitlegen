from pathlib import Path
from typing import Any

import numpy as np

from subtitlegen.visual.detection import text_detection_options, text_recognition_options
from subtitlegen.visual.models import BoundingBox
from subtitlegen.visual.ocr import OcrResult
from subtitlegen.visual.paddle_runtime import (
    PaddleWorkerClient,
    gpu_paddle_wheel_installed,
    should_start_paddle_worker,
    start_paddle_runtime,
)


def _fake_worker(requests: Any, responses: Any, device_type: str) -> None:
    while True:
        message = requests.get()
        operation = message.get("op")
        if operation == "stop":
            responses.put({"ok": True})
            return
        if operation == "ping":
            responses.put({"ok": True, "device": device_type})
        elif operation == "detect":
            responses.put(
                {
                    "ok": True,
                    "boxes": [{"x": 2, "y": 3, "width": 7, "height": 7, "score": 0.8}],
                }
            )
        elif operation == "detect_batch":
            responses.put(
                {
                    "ok": True,
                    "batches": [
                        [{"x": 2, "y": 3, "width": 7, "height": 7, "score": 0.8}]
                        for _ in message["images"]
                    ],
                }
            )
        elif operation == "recognize":
            responses.put({"ok": True, "text": "ドレスローザ"})
        else:
            responses.put({"ok": False, "error": operation})


def test_paddle_worker_module_does_not_import_torch() -> None:
    source = Path("src/subtitlegen/visual/paddle_worker.py").read_text(encoding="utf-8")
    imports = [
        line
        for line in source.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]
    joined = "\n".join(imports)
    assert "torch" not in joined
    assert "manga_ocr" not in joined
    assert "transformers" not in joined


def test_paddle_worker_client_round_trip() -> None:
    client = PaddleWorkerClient(device_type="gpu", worker=_fake_worker)
    try:
        assert client.ping()["device"] == "gpu"
        assert client.detect(np.zeros((8, 8), dtype=np.uint8)) == (
            BoundingBox(2, 3, 7, 7, 0.8),
        )
        batches = client.detect_batch(
            [np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8)]
        )
        assert len(batches) == 2
        assert client.recognize(np.zeros((8, 8), dtype=np.uint8)) == "ドレスローザ"
    finally:
        client.close()
        client.close()


def test_start_paddle_runtime_skips_without_gpu_wheel(monkeypatch: Any) -> None:
    monkeypatch.delenv("SUBTITLEGEN_PADDLE_DEVICE", raising=False)
    monkeypatch.setattr(
        "subtitlegen.visual.paddle_runtime.gpu_paddle_wheel_installed",
        lambda: False,
    )
    assert should_start_paddle_worker() is False
    assert start_paddle_runtime() is None


def test_start_paddle_runtime_falls_back_when_worker_fails(monkeypatch: Any) -> None:
    monkeypatch.setenv("SUBTITLEGEN_PADDLE_DEVICE", "gpu")

    class Boom:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("no gpu")

    monkeypatch.setattr("subtitlegen.visual.paddle_runtime.PaddleWorkerClient", Boom)
    assert start_paddle_runtime() is None


def test_gpu_wheel_probe_is_boolean() -> None:
    assert gpu_paddle_wheel_installed() in {True, False}


def test_detection_options_accept_gpu_device() -> None:
    assert text_detection_options(device_type="gpu")["engine_config"]["device_type"] == "gpu"
    assert text_recognition_options()["engine_config"]["device_type"] == "cpu"


def test_detector_and_recognizer_use_runtime() -> None:
    from subtitlegen.visual.detection import PaddleOcrDetector
    from subtitlegen.visual.ocr import PaddleTextRecognizer

    class Runtime:
        def detect(self, _image: Any) -> tuple[BoundingBox, ...]:
            return (BoundingBox(1, 1, 2, 2),)

        def detect_batch(self, images: Any) -> tuple[tuple[BoundingBox, ...], ...]:
            return tuple((BoundingBox(1, 1, 2, 2),) for _ in images)

        def recognize(self, _image: Any) -> str:
            return "錦"

        def close(self) -> None:
            return None

    runtime = Runtime()
    detector = PaddleOcrDetector(runtime=runtime)
    recognizer = PaddleTextRecognizer(runtime=runtime)
    assert detector.detect(np.zeros((4, 4))) == (BoundingBox(1, 1, 2, 2),)
    assert recognizer.recognize(np.zeros((4, 4))) == OcrResult("錦")
    detector.close()
