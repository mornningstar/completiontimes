import asyncio

global_concurrency = asyncio.Semaphore(100)