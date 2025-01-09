import asyncio
import json
import logging
import time

import aiohttp
import backoff
from aiohttp import ClientResponseError
from requests.utils import parse_header_links

from src.data_handling.async_database import AsyncDatabase


class APIConnectionAsync:
    RESULTS_PER_PAGE = 100

    def __init__(self, github_repo_username_title):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = None
        self.config = self.load_config()
        self.access_token = self.config.get('github_access_token', '')
        self.github_repo_username_title = github_repo_username_title
        self.get_commits_url = f'https://api.github.com/repos/{github_repo_username_title}/commits'
        self.get_contents_url = f'https://api.github.com/repos/{github_repo_username_title}/contents'
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

    @property
    def file_tracking_collection(self):
        return f'{self.collection_name}_file_tracking'

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
                          giveup=lambda e: hasattr(e, 'status') and e.status == 400
                          )
    async def make_request(self, url):
        async with self.semaphore:
            async with self.session.get(url, headers={'Authorization': f'token {self.access_token}'}) as response:
                if 'X-RateLimit-Remaining' in response.headers and response.headers['X-RateLimit-Remaining'] < '1':
                    reset_time = int(response.headers['X-RateLimit-Reset'])
                    sleep_time = reset_time - time.time() + 10  # Add a buffer of 10 seconds
                    print(f'Rate limit exceeded. Sleeping for {sleep_time} seconds.')
                    await asyncio.sleep(sleep_time)
                    return await self.make_request(url)

                response.raise_for_status()
                return await response.json(), response.headers

    async def get_next_link(self, headers):
        link_header = headers.get('link', None)

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

    """
    Objective: Calls GET api.github.com/repos/:username/:repository/commits/:sha' to receive full commit data.
    """
    async def get_commit_info(self, sha):
        get_commit_info_url = f'{self.get_commits_url}/{str(sha)}'
        response, headers = await self.make_request(get_commit_info_url)

        file_paths = set()
        for file in response['files']:
            file_paths.add(file['filename'])

        return response, file_paths

    async def batch_get_commit_info(self, batch_size=200):
        commit_info_list = []
        file_paths_set = set()
        file_path_data = []

        list_shas = await AsyncDatabase.fetch_all_shas(self.commit_list_collection)
        full_info_shas = await AsyncDatabase.fetch_all_shas(self.full_commit_info_collection)
        missing_shas = list_shas - full_info_shas



        for sha in missing_shas:
            commit_info, file_paths = await self.get_commit_info(sha)
            commit_info_list.append(commit_info)
            file_paths_set.update(file_paths)
            file_path_data = [{'path': file_path} for file_path in file_paths_set]

            if len(commit_info_list) >= batch_size:
                await AsyncDatabase.insert_many(self.full_commit_info_collection, commit_info_list, data_type='commit')
                await AsyncDatabase.insert_many(self.file_tracking_collection, file_path_data, data_type='files')
                commit_info_list.clear()
                file_paths_set.clear()

        if commit_info_list or file_paths_set:
            await AsyncDatabase.insert_many(self.full_commit_info_collection, commit_info_list, data_type='commit')
            await AsyncDatabase.insert_many(self.file_tracking_collection, file_path_data, data_type='files')

    async def get_file_size_at_commit(self, file_path, sha):
        url = f'{self.get_contents_url}/{file_path}?ref={sha}'

        try:
            response, _ = await self.make_request(url)
            return response['size']
        except ClientResponseError as e:
            if e.status == 404:
                return 'file_not_found'
            else:
                raise

    async def get_commit_details(self, sha):
        commit_details = await AsyncDatabase.find(self.full_commit_info_collection, {'sha': sha})
        return commit_details

    async def is_file_deleted_in_commit(self, commit_details, file_path):
        for commit_detail in commit_details:
            if 'files' in commit_detail:
                for file in commit_detail['files']:
                    if file['filename'] == file_path and file['status'] == 'removed':
                        return True
        return False

    async def get_file_commit_history(self, file_path):
        url = f'{self.get_commits_url}?path={file_path}&per_page={APIConnectionAsync.RESULTS_PER_PAGE}'

        while url:
            print("Fetching URL: ", url)  # Debugging statement
            commits, headers = await self.make_request(url)

            if not commits:
                print(f'No commit history has been found for {file_path}')
                break

            commit_history = []

            for commit in commits:
                sha = commit['sha']
                commit_date = commit['commit']['author']['date']
                size_or_status = await self.get_file_size_at_commit(file_path, sha)

                if size_or_status == 'file_not_found':
                    commit_details = await self.get_commit_details(commit['sha'])
                    if await self.is_file_deleted_in_commit(commit_details, file_path):
                        size_or_status = 0  # DELETED
                        print("DELETED")
                    else:
                        print("NOT DELETED.")

                commit_history.append({
                    'sha': sha,
                    'date': commit_date,
                    'size': size_or_status
                })

            data = {
                'path': file_path,
                'commit_history': commit_history
            }

            await AsyncDatabase.insert_many(self.file_tracking_collection, [data], data_type='files')

            # Get the next page URL from headers if available
            url = await self.get_next_link(headers)
            print("Next URL: ", url)  # Debugging statement

    async def iterate_over_file_paths(self):
        files = await AsyncDatabase.fetch_all(self.file_tracking_collection)

        for file in files:
            await self.get_file_commit_history(file['path'])

    async def populate_db(self):
        try:
            commit_list_exists = await AsyncDatabase.find_one(self.commit_list_collection, {})
            update = commit_list_exists is not None

            await self.get_commit_list(update=update)
            await self.batch_get_commit_info()
            await self.iterate_over_file_paths()

        except Exception as e:
            print("An error occurred in populate_db: " + str(e))

        finally:
            await self.close_session()
