import asyncio
import logging
import time
from collections import deque


class TokenBucket:
    def __init__(self, capacity: int = 900, window: int = 60):
        self.capacity = capacity
        self.window = window
        self.timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

        self.logger = logging.getLogger(self.__class__.__name__)

    async def acquire(self):
        async with self._lock:
            now = time.time()
            while self.timestamps and now - self.timestamps[0] > self.window:
                self.timestamps.popleft()
            
            if len(self.timestamps) < self.capacity:
                self.timestamps.append(now)
                return
            
            oldest = self.timestamps[0]
            sleep_for = self.window - (now - oldest) + 0.01
            self.logger.debug(f"Bucket full, sleeping {sleep_for:.2f}s")

        await asyncio.sleep(sleep_for)
