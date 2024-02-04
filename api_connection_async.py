import asyncio
import json
import time

import aiohttp
import backoff
from requests.utils import parse_header_links

from async_database import AsyncDatabase


class APIConnectionAsync:
    RESULTS_PER_PAGE = 100

    def __init__(self, github_repo_username_title):
        self.session = None
        self.config = self.load_config()
        self.access_token = self.config.get('github_access_token', '')
        self.github_repo_username_title = github_repo_username_title
        self.get_commits_url = f'https://api.github.com/repos/{github_repo_username_title}/commits'
        self.collection_name = self.github_repo_username_title.replace("/", "_")
        self.semaphore = asyncio.Semaphore(100)

    @classmethod
    async def create(cls, github_repo_username_title):
        self = APIConnectionAsync(github_repo_username_title)
        self.session = aiohttp.ClientSession()  # Create a session when an instance is created
        await AsyncDatabase.initialize()
        return self

    @property
    def commit_list_collection(self):
        return f'{self.collection_name}_commit_list'

    @property
    def full_commit_info_collection(self):
        return f"{self.collection_name}_full_commit_info"

    @staticmethod
    def load_config():
        try:
            with open('config.json', 'r') as config_file:
                return json.load(config_file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    async def close_session(self):
        await self.session.close()

    @backoff.on_exception(backoff.expo,
                          aiohttp.ClientError,
                          max_tries=4,
                          giveup=lambda e: e.status == 400)
    async def make_request(self, url):
        sleep_time = 0

        async with self.semaphore:
            async with self.session.get(url, headers={'Authorization': f'token {self.access_token}'}) as response:

                if 'X-RateLimit-Remaining' in response.headers and response.headers['X-RateLimit-Remaining'] < '1':
                    reset_time = int(response.headers['X-RateLimit-Reset'])
                    sleep_time = reset_time - time.time() + 10  # Add a buffer of 10 seconds
                    print(f'Rate limit exceeded. Sleeping for {sleep_time} seconds.')

                response.raise_for_status()

                if sleep_time <= 0:
                    try:
                        json_data = await response.json()
                        return json_data, response.headers
                    except aiohttp.ClientResponseError as e:
                        print(f'HTTP Error: {e}')
                        return None
                    except json.JSONDecodeError as e:
                        print(f'Failed to decode JSON: {e}')
                        return

        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
            return await self.make_request(url)

    async def get_next_link(self, headers):
        link_header = headers.get('link', None)
        # link_header = response.headers.get('link', None)

        if link_header:
            links = parse_header_links(link_header)
            next_link = [link['url'] for link in links if link['rel'] == 'next']
            return next_link[0] if next_link else None

        return None

    async def get_commit_list(self, update=False):
        url = self.get_commits_url + '?per_page=' + str(APIConnectionAsync.RESULTS_PER_PAGE)

        while url:
            response_json, headers = await self.make_request(url)

            if not response_json:
                break

            if update:
                new_commits = []
                for commit in response_json:
                    existing_commit = await AsyncDatabase.find_one(self.commit_list_collection, {'sha': commit['sha']})

                    if existing_commit:
                        if new_commits:
                            await AsyncDatabase.insert_many(self.commit_list_collection, response_json)

                        url = None
                        break
                    else:
                        new_commits.append(commit)

            else:
                await AsyncDatabase.insert_many(self.commit_list_collection, response_json)

            if url:
                url = await self.get_next_link(headers)

    async def get_commit_info(self, sha):
        get_commit_info_url = f'{self.get_commits_url}/{str(sha)}'
        response, headers = await self.make_request(get_commit_info_url)

        return response

    async def batch_get_commit_info(self, update=False, batch_size=20):
        commit_info_list = []
        new_commits = []

        repository_commits = await AsyncDatabase.find(self.commit_list_collection, {})

        list_shas = await AsyncDatabase.fetch_all_shas(self.commit_list_collection)
        full_info_shas = await AsyncDatabase.fetch_all_shas(self.full_commit_info_collection)

        missing_shas = list_shas - full_info_shas

        for sha in missing_shas:
            commit_info = await self.get_commit_info(sha)
            commit_info_list.append(commit_info)

            if len(commit_info_list) >= batch_size:
                await AsyncDatabase.insert_many(self.full_commit_info_collection, commit_info_list)
                commit_info_list.clear()

        if commit_info_list:
            await AsyncDatabase.insert_many(self.full_commit_info_collection, commit_info_list)

    async def populate_db(self):
        try:
            commit_list_exists = await AsyncDatabase.find_one(self.commit_list_collection, {})
            update = commit_list_exists is not None

            await self.get_commit_list(update=update)
            await self.batch_get_commit_info(update=update)

        except Exception as e:
            print("An error ocurred in populate_db: " + str(e))

        finally:
            await self.close_session()
