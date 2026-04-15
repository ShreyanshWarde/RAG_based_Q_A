from collections import defaultdict, deque
from threading import Lock
from time import time


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: defaultdict[str, deque[float]] = defaultdict(deque)
        self.lock = Lock()

    def check(self, identifier: str) -> None:
        with self.lock:
            now = time()
            request_times = self.requests[identifier]

            while request_times and now - request_times[0] > self.window_seconds:
                request_times.popleft()

            if len(request_times) >= self.max_requests:
                raise ValueError("Rate limit exceeded. Please try again later.")

            request_times.append(now)
