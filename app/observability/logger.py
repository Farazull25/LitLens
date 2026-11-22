"""Structured JSON logging and per-step execution tracing."""

from __future__ import annotations

import contextvars
import logging
import sys
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import structlog

_trace_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
_step_traces_ctx: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "step_traces", default=None
)


@dataclass
class StepTrace:
    """Record of a single agent or RAG pipeline step."""

    trace_id: str
    step_name: str
    status: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


def configure_logging() -> None:
    """Configure structlog to emit structured JSON logs suitable for CloudWatch."""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "litlens") -> structlog.stdlib.BoundLogger:
    """Get a structured logger bound with current trace context."""
    return structlog.get_logger(name)


def start_trace(trace_id: str | None = None) -> str:
    """Initialize a new request trace ID and clear step buffers."""
    tid = trace_id or uuid.uuid4().hex[:12]
    _trace_id_ctx.set(tid)
    _step_traces_ctx.set([])
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id=tid)
    return tid


def get_current_trace_id() -> str:
    """Return active trace ID or generate one."""
    tid = _trace_id_ctx.get()
    if not tid:
        tid = start_trace()
    return tid


def get_step_traces() -> list[dict[str, Any]]:
    """Return list of recorded steps for the current trace."""
    return list(_step_traces_ctx.get() or [])


@contextmanager
def trace_step(step_name: str, **metadata: Any) -> Generator[dict[str, Any], None, None]:
    """Context manager measuring latency and recording structured trace events."""
    logger = get_logger()
    tid = get_current_trace_id()
    start = time.perf_counter()
    step_meta: dict[str, Any] = dict(metadata)
    status = "ok"
    try:
        yield step_meta
    except Exception as exc:
        status = "error"
        step_meta["error"] = str(exc)
        raise
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        record = StepTrace(
            trace_id=tid,
            step_name=step_name,
            status=status,
            latency_ms=latency_ms,
            metadata=step_meta,
        )
        traces = list(_step_traces_ctx.get() or [])
        traces.append(asdict(record))
        _step_traces_ctx.set(traces)
        logger.info(
            "step_completed",
            step_name=step_name,
            status=status,
            latency_ms=latency_ms,
            **step_meta,
        )
