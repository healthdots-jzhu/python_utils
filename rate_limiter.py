"""Reusable in-memory rate limiting helpers.

The limiter is framework-agnostic and exposes a dependency factory that can be
plugged into FastAPI without requiring FastAPI at import time.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Callable, Deque, Mapping, MutableMapping


RequestKeyFunc = Callable[[Any], str]


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    reset_after: int
    retry_after: int
    key: str


class RateLimitExceeded(Exception):
    """Raised when a request exceeds the configured rate limit."""

    def __init__(self, message: str, headers: Mapping[str, str]):
        super().__init__(message)
        self.headers = dict(headers)


def default_request_key(request: Any) -> str:
    """Build a stable key from client identity and request path."""
    headers = getattr(request, "headers", {}) or {}
    forwarded_for = headers.get("x-forwarded-for", "")
    client_ip = forwarded_for.split(",", 1)[0].strip()
    if not client_ip:
        client = getattr(request, "client", None)
        client_ip = getattr(client, "host", "") or "global"

    method = getattr(request, "method", "*") or "*"
    url = getattr(request, "url", None)
    path = getattr(url, "path", "*") if url is not None else "*"
    return f"{client_ip}:{method.upper()}:{path}"


class InMemoryRateLimiter:
    """Fixed-window in-memory rate limiter.

    Example:

        limiter = InMemoryRateLimiter(limit=100, window_seconds=60)
        dependency = limiter.as_dependency()
    """

    def __init__(
        self,
        limit: int,
        window_seconds: int,
        key_func: RequestKeyFunc | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if limit <= 0:
            raise ValueError("limit must be greater than 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than 0")

        self.limit = limit
        self.window_seconds = window_seconds
        self.key_func = key_func or default_request_key
        self._clock = clock or time.monotonic
        self._events: MutableMapping[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, bucket: Deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

    def _reset_after(self, bucket: Deque[float], now: float) -> int:
        if not bucket:
            return self.window_seconds
        remaining_window = self.window_seconds - (now - bucket[0])
        return max(0, math.ceil(remaining_window))

    def check(self, key: str) -> RateLimitResult:
        with self._lock:
            now = self._clock()
            bucket = self._events[key]
            self._prune(bucket, now)
            current = len(bucket)
            allowed = current < self.limit
            remaining = max(0, self.limit - current - (1 if allowed else 0))
            reset_after = self._reset_after(bucket, now)
            retry_after = 0 if allowed else reset_after
            return RateLimitResult(
                allowed=allowed,
                limit=self.limit,
                remaining=remaining,
                reset_after=reset_after,
                retry_after=retry_after,
                key=key,
            )

    def hit(self, key: str) -> RateLimitResult:
        with self._lock:
            now = self._clock()
            bucket = self._events[key]
            self._prune(bucket, now)

            if len(bucket) >= self.limit:
                reset_after = self._reset_after(bucket, now)
                return RateLimitResult(
                    allowed=False,
                    limit=self.limit,
                    remaining=0,
                    reset_after=reset_after,
                    retry_after=reset_after,
                    key=key,
                )

            bucket.append(now)
            reset_after = self._reset_after(bucket, now)
            return RateLimitResult(
                allowed=True,
                limit=self.limit,
                remaining=max(0, self.limit - len(bucket)),
                reset_after=reset_after,
                retry_after=0,
                key=key,
            )

    def headers_for(self, result: RateLimitResult) -> dict[str, str]:
        return {
            "X-RateLimit-Limit": str(result.limit),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(result.reset_after),
            "Retry-After": str(result.retry_after),
        }

    def as_dependency(self, key_func: RequestKeyFunc | None = None) -> Callable[..., Any]:
        resolver = key_func or self.key_func

        async def dependency(request: Any, response: Any = None) -> RateLimitResult:
            key = resolver(request)
            result = self.hit(key)
            headers = self.headers_for(result)

            if response is not None and hasattr(response, "headers"):
                response.headers.update(headers)

            if not result.allowed:
                self._raise_limit_error(headers)

            return result

        return dependency

    def _raise_limit_error(self, headers: Mapping[str, str]) -> None:
        try:
            from fastapi import HTTPException, status  # type: ignore
        except ImportError:
            raise RateLimitExceeded("Rate limit exceeded", headers)

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers=dict(headers),
        )