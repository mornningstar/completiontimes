import asyncio
import logging
import random
import time
from datetime import datetime

import backoff
from aiohttp import ClientSession, ClientResponse, ClientError, ClientResponseError, ClientTimeout
from requests.utils import parse_header_links

from src.github.token_bucket import TokenBucket


def log_backoff_success(details):
    logger = logging.getLogger("GitHubClient")
    tries = details["tries"]
    seconds = details["elapsed"]
    url = details["args"][1] if len(details["args"]) > 1 else "?"

    if tries > 1:
        logger.info(f"Backoff succeeded after {tries} tries ({seconds:.2f}s) on URL: {url}")

class GitHubClient:
    """ Wrapper around aiohttp to be able to talk to the GitHub API. """

    def __init__(self, auth_token: str, token_bucket: TokenBucket = None, concurrency: int = 100, timeout: int = 300,
                 max_retries: int = 5):
        self.auth_token = auth_token
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.max_retires = max_retries
        self._session = None
        self._headers = {
            "Authorization": f"token {auth_token}",
        }
        self.bucket = token_bucket if token_bucket else TokenBucket()
        self.rate_limit_lock = asyncio.Lock()
        self.rate_limit_reset_time = 0

        self.logger = logging.getLogger(self.__class__.__name__)

    async def open(self):
        timeout = ClientTimeout(total=self.timeout,      # entire request must finish in 5 mins
                                sock_read=60)   # 1 minute allowance to read response body
        self._session = ClientSession(timeout=timeout)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    @backoff.on_exception(backoff.expo,
                          (ClientError, asyncio.TimeoutError),
                          max_tries=5,
                          giveup=lambda e: isinstance(e, ClientResponseError) and not GitHubClient._should_retry(e),
                          on_success=log_backoff_success
                          )
    async def get(self, url):
        """ Call GET to URL while respecting the rate limit """
        async with self.semaphore:
            await self.bucket.acquire()
            
            try:
                async with self._session.get(url, headers=self._headers) as response:
                    await self._respect_rate_limit(response)
                    response.raise_for_status()
                    data = await response.json()
                    return data, response.headers
            except ClientResponseError as e:
                if self._secondary_rate_limited(e):
                    retry_after = float(e.headers.get("Retry-After", 60))
                    jitter = random.uniform(1.0, 5.0)
                    wait_time = retry_after + jitter
                    self.logger.warning(f"Secondary rate limit hit. Sleeping for {wait_time:.2f} seconds. URL: {url}")

                    await asyncio.sleep(wait_time)
                    raise
                elif e.status == 403:
                    await self._respect_rate_limit(e)
                    raise
                raise

    @staticmethod
    def _should_retry(err: ClientResponseError) -> bool:
        """ Allow a retry on 429, 403 with 0 remaining and on classic 5xx """
        if err.status in (500, 502, 503, 429):
            return True
        if err.status == 403 and int(err.headers.get("X-RateLimit-Remaining", "1")) == 0:
            return True # primary rate limit
        if GitHubClient._secondary_rate_limited(err):
            return True  # secondary rate limit
        return False

    @staticmethod
    def _secondary_rate_limited(err: ClientResponseError) -> bool:
        body = err.message.lower() if isinstance(err.message, str) else ""
        return (
                err.status == 403
                and "secondary rate limit" in body
        )

    async def _respect_rate_limit(self, response: ClientResponse | ClientResponseError):
        remaining = int(response.headers.get('X-RateLimit-Remaining', '1'))
        if remaining > 0:
            return

        header_reset = float(response.headers['X-RateLimit-Reset']) + 3.0

        async with self.rate_limit_lock:
            if header_reset > self.rate_limit_reset_time:
                self.rate_limit_reset_time = header_reset

                sleep_time = max(0.0, self.rate_limit_reset_time - time.time())
                if sleep_time:
                    local_reset_time = datetime.fromtimestamp(self.rate_limit_reset_time)
                    self.logger.info(f'Rate limit exceeded. Sleeping for {sleep_time} seconds until {local_reset_time}.')
                    await asyncio.sleep(sleep_time)

                    cooldown = random.uniform(1.0, 5.0)
                    self.logger.info(f"Cooldown after reset: sleeping extra {cooldown:.2f} seconds.")
                    await asyncio.sleep(cooldown)

        sleep_time = max(0.0, self.rate_limit_reset_time - time.time())
        if sleep_time:
            await asyncio.sleep(sleep_time)

    async def paginate(self, url):
        while url:
            response_json, headers = await self.get(url)

            if not response_json:
                break

            yield response_json, headers
            url = self.get_next_link(headers)

    def get_next_link(self, headers):
        link_header = headers.get('link')

        if not link_header:
            return None

        links = parse_header_links(link_header)
        next_link = [link['url'] for link in links if link['rel'] == 'next']

        if not next_link:
            return None

        return next_link[0]
