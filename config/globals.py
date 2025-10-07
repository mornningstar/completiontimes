import asyncio
import os

global_concurrency = asyncio.Semaphore(100)
CPU_LIMIT = int(os.getenv('MAX_CPUS', 36))