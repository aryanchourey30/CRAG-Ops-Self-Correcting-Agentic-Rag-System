"""Pipeline trace helpers."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from app.observability.logger import serialize_for_log


def new_trace_id() -> str:
    """Generate a unique trace identifier."""

    return uuid.uuid4().hex


def append_step_log(
    state: dict[str, Any],
    *,
    step: str,
    input_data: Any,
    output_data: Any,
    decision: str,
) -> None:
    """Append a structured log entry into graph state."""

    state.setdefault("logs", []).append(
        {
            "step": step,
            "input": serialize_for_log(input_data),
            "output": serialize_for_log(output_data),
            "decision": decision,
        }
    )


def persist_trace(log_dir: Path, trace_id: str, payload: dict[str, Any]) -> None:
    """Persist the full trace payload to disk."""

    trace_dir = log_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{trace_id}.json"
    trace_path.write_text(json.dumps(serialize_for_log(payload), indent=2), encoding="utf-8")
