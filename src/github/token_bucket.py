import asyncio
import time
from collections import deque


class TokenBucket:
    def __init__(self):
        self.capacity = 900 # max. 900 points
        self.window = 60 # per 60 seconds
        self.timestamps: deque[float] = deque() # keep time of last N acquisitions
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.time()
            # drop tokens that slid out of the window
            while self.timestamps and now - self.timestamps[0] > self.window:
                self.timestamps.popleft()
            
            if len(self.timestamps) >= self.capacity:
                # wait until at least one token slides out
                sleep_for = self.window - (now - self.timestamps[0]) + 0.01
                await asyncio.sleep(sleep_for)
            
            self.timestamps.append(time.time())