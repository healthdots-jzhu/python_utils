import asyncio
import unittest

from rate_limiter import InMemoryRateLimiter, RateLimitExceeded, default_request_key


class FakeClock:
    def __init__(self) -> None:
        self.current = 0.0

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


class DummyClient:
    def __init__(self, host: str) -> None:
        self.host = host


class DummyUrl:
    def __init__(self, path: str) -> None:
        self.path = path


class DummyRequest:
    def __init__(self, host: str = "127.0.0.1", path: str = "/items", method: str = "GET", headers=None) -> None:
        self.client = DummyClient(host)
        self.url = DummyUrl(path)
        self.method = method
        self.headers = headers or {}


class DummyResponse:
    def __init__(self) -> None:
        self.headers = {}


class RateLimiterTest(unittest.TestCase):
    def test_hit_blocks_after_limit_and_recovers_after_window(self):
        clock = FakeClock()
        limiter = InMemoryRateLimiter(limit=2, window_seconds=10, clock=clock)

        first = limiter.hit("client-1")
        second = limiter.hit("client-1")
        blocked = limiter.hit("client-1")

        self.assertTrue(first.allowed)
        self.assertEqual(first.remaining, 1)
        self.assertTrue(second.allowed)
        self.assertEqual(second.remaining, 0)
        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.retry_after, 10)

        clock.advance(10)
        recovered = limiter.hit("client-1")
        self.assertTrue(recovered.allowed)
        self.assertEqual(recovered.remaining, 1)

    def test_default_request_key_uses_forwarded_ip_method_and_path(self):
        request = DummyRequest(
            host="10.0.0.8",
            path="/v1/providers",
            method="post",
            headers={"x-forwarded-for": "203.0.113.10, 10.0.0.8"},
        )

        self.assertEqual(default_request_key(request), "203.0.113.10:POST:/v1/providers")

    def test_dependency_sets_headers_and_raises_on_limit(self):
        clock = FakeClock()
        limiter = InMemoryRateLimiter(limit=1, window_seconds=30, clock=clock)
        dependency = limiter.as_dependency()
        request = DummyRequest(path="/limited")

        response = DummyResponse()
        first = asyncio.run(dependency(request, response))
        self.assertTrue(first.allowed)
        self.assertEqual(response.headers["X-RateLimit-Remaining"], "0")

        with self.assertRaises(Exception) as ctx:
            asyncio.run(dependency(request, DummyResponse()))

        exc = ctx.exception
        if isinstance(exc, RateLimitExceeded):
            self.assertEqual(exc.headers["Retry-After"], "30")
        else:
            self.assertEqual(getattr(exc, "status_code", None), 429)
            self.assertEqual(getattr(exc, "headers", {}).get("Retry-After"), "30")


if __name__ == "__main__":
    unittest.main()