import asyncio
import json
import logging
import time
from datetime import datetime

import backoff
from aiohttp import ClientSession, ClientResponse, ClientError, ClientResponseError
from dateutil.tz import tzlocal
from requests.utils import parse_header_links

from src.github.token_bucket import TokenBucket


def _backoff_handler(details):
    err = details["exception"]
    if isinstance(err, ClientResponseError) and "Retry-After" in err.headers:
        return float(err.headers["Retry-After"])

class GitHubClient:
    BASE = "https://api.github.com/repos/"

    def __init__(self, auth_token: str, concurrency: int = 100):
        self.auth_token = auth_token
        self.semaphore = asyncio.Semaphore(concurrency)
        self._session = None
        self._headers = {
            "Authorization": f"token {auth_token}",
        }
        self.bucket = TokenBucket()

        self.logger = logging.getLogger(self.__class__.__name__)

    async def open(self):
        self._session = ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    @backoff.on_exception(backoff.expo,
                          ClientError,
                          max_tries=4,
                          giveup=lambda e: isinstance(e, ClientResponseError) and not GitHubClient._should_retry(e),
                          on_backoff=_backoff_handler,
                          )
    async def get(self, url):
        """ Call GET to URL while respecting the rate limit """
        async with self.semaphore:
            await self.bucket.acquire()

            async with self._session.get(url, headers=self._headers) as response:
                await self._respect_rate_limit(response)
                response.raise_for_status()
                data = await response.json()
                return data, response.headers

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
        if err.status != 403:
            return False
        # GitHub puts the explanation in the JSON body
        try:
            body = json.loads(err.message) if isinstance(err.message, str) else {}
            return "secondary rate limit" in (body.get("message") or "").lower()
        except Exception:
            return False

    async def _respect_rate_limit(self, response: ClientResponse):
        remaining = int(response.headers.get('X-RateLimit-Remaining', '1'))
        if remaining > 0:
            return

        reset_time = int(response.headers['X-RateLimit-Reset'])
        local_reset_time = datetime.fromtimestamp(reset_time)
        sleep_time = reset_time - time.time() + 2  # Add a buffer of 2 seconds
        self.logger.info(f'Rate limit exceeded. Sleeping for {sleep_time} seconds until {local_reset_time}.')

        await asyncio.sleep(sleep_time)

    async def paginate(self, url):
        while url:
            response_json, headers = await self.get(url)

            if not response_json:
                break

            yield response_json, headers
            url = await self.get_next_link(headers)

    async def get_next_link(self, headers):
        link_header = headers.get('link')

        if not link_header:
            self.logger.debug("No 'link' header found in response.")
            return None

        links = parse_header_links(link_header)
        next_link = [link['url'] for link in links if link['rel'] == 'next']

        if not next_link:
            self.logger.debug("No 'next' link found in 'link' header.")
            return None

        return next_link[0]
