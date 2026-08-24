from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _HuggingFaceTokenizerFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "use_fast" not in record.getMessage()


def _silence_huggingface_tokenizer_warning() -> None:
    """Manga OCR's tokenizer has no Rust fast path; the fallback is expected."""
    warnings.filterwarnings(
        "ignore",
        message=r".*use_fast.*fast version.*",
    )
    token_filter = _HuggingFaceTokenizerFilter()
    for name in (
        "transformers",
        "transformers.tokenization_utils_base",
        "transformers.tokenization_auto",
    ):
        logging.getLogger(name).addFilter(token_filter)


def configure_logging(verbose: bool = False) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    _silence_huggingface_tokenizer_warning()
