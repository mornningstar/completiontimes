import asyncio

global_concurrency = asyncio.Semaphore(100)
CPU_LIMIT = int(36)

# Heuristic for stable line change
STABILITY_MIN_COMMITS = 3
STABILITY_MIN_DAYS = 14
STABILITY_IDLE_DAYS = 30

STABILITY_PERCENTAGE_CHANGE = 0.15