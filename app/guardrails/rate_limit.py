"""Sliding-window rate limiter protecting the 2-vCPU Space and free-tier quotas."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock


class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding window rate limiter."""

    def __init__(self, max_requests_per_minute: int = 20) -> None:
        self.max_requests = max_requests_per_minute
        self.window_seconds = 60.0
        self._lock = Lock()
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client_key: str = "global") -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            q = self._requests[client_key]
            while q and (now - q[0]) > self.window_seconds:
                q.popleft()
            if len(q) >= self.max_requests:
                retry_after = round(self.window_seconds - (now - q[0]), 2)
                return False, max(retry_after, 0.1)
            q.append(now)
            return True, 0.0

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
