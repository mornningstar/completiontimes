import asyncio
import logging
import time

import backoff
from aiohttp import ClientSession, ClientResponse, ClientError
from requests.utils import parse_header_links


class GitHubClient:
    BASE = "https://api.github.com/repos/"

    def __init__(self, auth_token: str, concurrency: int = 100):
        self.auth_token = auth_token
        self.semaphore = asyncio.Semaphore(concurrency)
        self._session = None
        self._headers = {
            "Authorization": f"token {auth_token}",
        }

        self.logger = logging.getLogger(self.__class__.__name__)

    async def open(self):
        self._session = ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    @backoff.on_exception(backoff.expo,
                          ClientError,
                          max_tries=4,
                          giveup=lambda e: hasattr(e, 'status') and e.status == 400
                          )
    async def get(self, url):
        async with self.semaphore:
            async with self._session.get(url, headers=self._headers) as response:
                response.raise_for_status()
                data = await response.json()
                await self._respect_rate_limit(response)

                return data, response.headers

    async def _respect_rate_limit(self, response: ClientResponse):
        remaining = int(response.headers['X-RateLimit-Remaining'])
        if remaining > 0:
            return

        reset_time = int(response.headers['X-RateLimit-Reset'])
        sleep_time = reset_time - time.time() + 5  # Add a buffer of 5 seconds
        self.logger.info(f'Rate limit exceeded. Sleeping for {sleep_time} seconds.')
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
