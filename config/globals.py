import asyncio

global_concurrency = asyncio.Semaphore(100)
CPU_LIMIT = int(36)