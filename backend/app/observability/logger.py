"""Application logging helpers."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


def _build_handler(path: Path) -> RotatingFileHandler:
    handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    return handler


def configure_logging(log_dir: Path) -> None:
    """Configure file-based logging once for the application."""

    log_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("app.log", "pipeline.log", "error.log"):
        (log_dir / filename).touch(exist_ok=True)

    root = logging.getLogger("crag_ops")
    if root.handlers:
        return

    root.setLevel(logging.INFO)
    root.addHandler(_build_handler(log_dir / "app.log"))

    pipeline = logging.getLogger("crag_ops.pipeline")
    pipeline.setLevel(logging.INFO)
    pipeline.addHandler(_build_handler(log_dir / "pipeline.log"))
    pipeline.propagate = False

    error_logger = logging.getLogger("crag_ops.error")
    error_logger.setLevel(logging.ERROR)
    error_logger.addHandler(_build_handler(log_dir / "error.log"))
    error_logger.propagate = False


def serialize_for_log(value: Any) -> Any:
    """Convert complex values into JSON-serializable structures."""

    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {key: serialize_for_log(item) for key, item in value.items()}
        if isinstance(value, list):
            return [serialize_for_log(item) for item in value]
        return str(value)
