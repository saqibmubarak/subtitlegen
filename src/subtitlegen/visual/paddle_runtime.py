from __future__ import annotations

import importlib.metadata
import logging
import os
from collections.abc import Sequence
from multiprocessing import get_context
from typing import Any

import numpy as np

from subtitlegen.visual.models import BoundingBox

logger = logging.getLogger(__name__)


def gpu_paddle_wheel_installed() -> bool:
    for distribution in importlib.metadata.distributions():
        name = (distribution.metadata.get("Name") or "").casefold()
        if name == "paddlepaddle-gpu":
            return True
    return False


def should_start_paddle_worker() -> bool:
    requested = os.environ.get("SUBTITLEGEN_PADDLE_DEVICE", "auto").strip().casefold()
    if requested == "cpu":
        return False
    if requested == "gpu":
        return True
    return gpu_paddle_wheel_installed()


def start_paddle_runtime() -> PaddleWorkerClient | None:
    if not should_start_paddle_worker():
        logger.info("paddle worker skipped; using in-process CPU Paddle")
        return None
    try:
        client = PaddleWorkerClient(device_type="gpu")
        client.ping()
        logger.info("paddle worker ready device=gpu")
        return client
    except Exception as error:
        logger.warning("paddle worker failed; using in-process CPU Paddle: %s", error)
        return None


class PaddleWorkerClient:
    """One-at-a-time Paddle detect/rec in a process that never imports torch."""

    def __init__(
        self,
        *,
        device_type: str = "gpu",
        worker: Any | None = None,
        context: Any | None = None,
    ) -> None:
        from subtitlegen.visual.paddle_worker import run_worker

        ctx = context or get_context("spawn")
        self._requests = ctx.Queue()
        self._responses = ctx.Queue()
        self._process = ctx.Process(
            target=worker or run_worker,
            args=(self._requests, self._responses, device_type),
            name="paddle-worker",
            daemon=True,
        )
        self._closed = False
        self._process.start()

    def ping(self) -> dict[str, Any]:
        return self._call({"op": "ping"})

    def detect(self, image: Any) -> tuple[BoundingBox, ...]:
        payload = self._call({"op": "detect", "image": np.asarray(image)})
        return _boxes_from_payload(payload.get("boxes", ()))

    def detect_batch(self, images: Sequence[Any]) -> tuple[tuple[BoundingBox, ...], ...]:
        payload = self._call(
            {"op": "detect_batch", "images": [np.asarray(image) for image in images]}
        )
        return tuple(_boxes_from_payload(item) for item in payload.get("batches", ()))

    def recognize(self, image: Any) -> str:
        payload = self._call({"op": "recognize", "image": np.asarray(image)})
        return str(payload.get("text") or "")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._requests.put({"op": "stop"}, timeout=2)
            self._responses.get(timeout=5)
        except Exception:
            pass
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=5)

    def _call(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("paddle worker is closed")
        self._requests.put(message)
        payload = self._responses.get()
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error") or "paddle worker failed")
        return payload


def _boxes_from_payload(items: Any) -> tuple[BoundingBox, ...]:
    boxes: list[BoundingBox] = []
    for item in items:
        boxes.append(
            BoundingBox(
                int(item["x"]),
                int(item["y"]),
                int(item["width"]),
                int(item["height"]),
                float(item.get("score", 1.0)),
            )
        )
    return tuple(boxes)
