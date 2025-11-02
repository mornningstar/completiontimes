import asyncio

global_concurrency = asyncio.Semaphore(100)
CPU_LIMIT = int(36)

# Heuristic for stable line change
MATURITY_MIN_COMMITS = 20
MATURITY_MIN_AGE_DAYS = 90

STABILITY_ABS_THRESHOLD_LINES = 20
STABILITY_MIN_COMMITS = 3