"""
Standard Python .log text logging (stdout and durable .log files).
Outputs human-readable log lines formatted as:
    YYYY-MM-DD HH:MM:SS,mmm [LEVEL] [component] [task_id] message | event=... [counters/details]
Buffers logs per task so they can be written as .log files to runs/{div}/{quarter}/{run_id}/logs/{task_id}.log.
"""
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_CTX: Dict[str, Any] = {}
_CTX_KEYS = ("run_id", "component", "task_id", "target", "fold_id", "attempt", "image")
_TASK_BUFFER: List[str] = []


def bind_context(
    run_id: str,
    component: str,
    task_id: str,
    target: Optional[str] = None,
    fold_id: Optional[str] = None,
    attempt: int = 1,
) -> None:
    """Bind process-wide context and reset task log buffer."""
    global _CTX, _TASK_BUFFER
    _CTX = {
        "run_id": run_id,
        "component": component,
        "task_id": task_id,
        "target": target,
        "fold_id": fold_id,
        "attempt": attempt,
        "image": os.environ.get("IMAGE_TAG", "local"),
    }
    _TASK_BUFFER = []


def get_context() -> Dict[str, Any]:
    return dict(_CTX)


def get_task_log_text() -> str:
    """Return all buffered log lines for the current task."""
    return "\n".join(_TASK_BUFFER) + ("\n" if _TASK_BUFFER else "")


def log_event(
    event: str,
    msg: str,
    level: str = "INFO",
    counters: Optional[Dict[str, Any]] = None,
    error_category: Optional[str] = None,
    duration_ms: Optional[float] = None,
    exc: Optional[str] = None,
    **fields: Any,
) -> None:
    """Emit JSON to stdout for log ingestion/tests and buffer standard Python .log text for file writing."""
    global _TASK_BUFFER
    now_utc = datetime.now(timezone.utc)
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    lvl = level.upper()
    comp = _CTX.get("component", "pipeline")
    tid = _CTX.get("task_id", "task")

    # 1. Standard Python .log line for disk/blob log file
    extras = [f"event={event}"]
    if error_category:
        extras.append(f"error_category={error_category}")
    if duration_ms is not None:
        extras.append(f"duration_ms={duration_ms}")
    if counters:
        extras.append(f"counters={counters}")
    for k, v in fields.items():
        extras.append(f"{k}={v}")

    extra_str = (" | " + " ".join(extras)) if extras else ""
    log_text_line = f"{now_str} [{lvl}] [{comp}] [{tid}] {msg}{extra_str}"
    if exc:
        log_text_line += f"\n{exc}"
    _TASK_BUFFER.append(log_text_line)

    # 2. Structured JSON record for stdout / container log ingestion
    record = {
        "timestamp": now_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "level": lvl,
        "event": event,
        "message": msg,
        "component": _CTX.get("component"),
        "task_id": _CTX.get("task_id"),
        "run_id": _CTX.get("run_id"),
        "target": _CTX.get("target"),
        "fold_id": _CTX.get("fold_id"),
        "attempt": _CTX.get("attempt", 1),
        "image": _CTX.get("image", "local"),
    }
    if error_category:
        record["error_category"] = error_category
    if duration_ms is not None:
        record["duration_ms"] = duration_ms
    if counters:
        record["counters"] = counters
    if exc:
        record["exc"] = exc
    for k, v in fields.items():
        record[k] = v

    import json
    sys.stdout.write(json.dumps(record, default=str) + "\n")
    sys.stdout.flush()


def info(msg: str, event: str = "PROGRESS", counters: Optional[Dict[str, Any]] = None, **fields: Any) -> None:
    log_event(event=event, msg=msg, level="INFO", counters=counters, **fields)


def warning(msg: str, event: str = "WARNING", error_category: Optional[str] = None, **fields: Any) -> None:
    log_event(event=event, msg=msg, level="WARNING", error_category=error_category, **fields)


def error(msg: str, event: str = "FAILED", error_category: Optional[str] = "UNKNOWN",
          exc: Optional[str] = None, **fields: Any) -> None:
    log_event(event=event, msg=msg, level="ERROR", error_category=error_category, exc=exc, **fields)


def log_provenance(
    source_run_id: Optional[str],
    path: str,
    target: Optional[str] = None,
    fingerprint: Optional[str] = None,
    msg: Optional[str] = None,
    event: str = "INPUT_LOADED",
    **fields: Any,
) -> None:
    """Log input provenance at point of use (source_run_id, path, target, fingerprint)."""
    message = msg or f"loaded input: {path}"
    log_event(
        event=event,
        msg=message,
        level="INFO",
        source_run_id=source_run_id,
        path=path,
        target=target or _CTX.get("target"),
        fingerprint=fingerprint,
        **fields,
    )

