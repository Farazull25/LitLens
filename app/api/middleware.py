"""FastAPI middleware for sliding-window rate limiting and request trace ID correlation."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.guardrails.rate_limit import SlidingWindowRateLimiter
from app.observability.logger import start_trace

_rate_limiter = SlidingWindowRateLimiter(
    max_requests_per_minute=get_settings().rate_limit_requests_per_minute
)


def get_rate_limiter() -> SlidingWindowRateLimiter:
    return _rate_limiter


class RateLimitAndTraceMiddleware(BaseHTTPMiddleware):
    """Enforces per-client sliding-window rate limiting and attaches X-Trace-Id headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        trace_id = start_trace(request.headers.get("X-Trace-Id"))
        if request.url.path.startswith("/api/"):
            client_ip = request.client.host if request.client else "anonymous"
            allowed, retry_after = _rate_limiter.allow(client_ip)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded. Please wait before sending more queries.",
                        "retry_after_seconds": retry_after,
                        "trace_id": trace_id,
                    },
                    headers={"Retry-After": str(int(retry_after) + 1), "X-Trace-Id": trace_id},
                )

        response: Response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response
