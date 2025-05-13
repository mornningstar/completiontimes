import logging

from aiohttp import ClientResponseError

from config.config import CONFIG
from src.data_handling.database.async_database import AsyncDatabase
from src.data_handling.database.commit_repo import CommitRepository
from src.github.http_client import GitHubClient


class APIConnectionAsync:
    RESULTS_PER_PAGE = 100

    def __init__(self, github_repo_username_title):
        self.logger = logging.getLogger(self.__class__.__name__)

        config = CONFIG[0]['github_access_token']
        self.access_token = config
        self.http_client = None
        self.commit_repo = None

        self.github_repo_username_title = github_repo_username_title
        self.get_commits_url = f'https://api.github.com/repos/{github_repo_username_title}/commits'
        self.get_contents_url = f'https://api.github.com/repos/{github_repo_username_title}/contents'
        self.collection_name = self.github_repo_username_title.replace("/", "_")

    @classmethod
    async def create(cls, github_repo_username_title):
        self = APIConnectionAsync(github_repo_username_title)
        self.http_client = GitHubClient(self.access_token)
        self.commit_repo = CommitRepository(self.github_repo_username_title)
        await self.http_client.open()
        await AsyncDatabase.initialize()

        return self

    @property
    def full_commit_info_collection(self):
        return f"{self.collection_name}_full_commit_info"

    @property
    def file_tracking_collection(self):
        return f'{self.collection_name}_file_tracking'

    async def close_session(self):
        await self.http_client.close()

    async def get_commit_list(self, update=False):
        self.logger.info(f'Getting commit list for {self.github_repo_username_title}')

        url = self.get_commits_url + '?per_page=' + str(APIConnectionAsync.RESULTS_PER_PAGE) # to http_client!

        async for page, headers in self.http_client.paginate(url):
            if update:
                new_commits = []
                for commit in page:
                    if not await self.commit_repo.find_commit(commit['sha'], full=False):
                        new_commits.append(commit)

                    await self.commit_repo.insert_new_commits(new_commits, full=False)
                    
                    if not new_commits:
                        self.logger.info("No new commits found on this page, stopping pagination.")
                        break

            else:
                await self.commit_repo.insert_new_commits(page, full=False)


    """
    Objective: Calls GET api.github.com/repos/:username/:repository/commits/:sha' to receive full commit data.
    """
    async def get_commit_info(self, sha):
        get_commit_info_url = f'{self.get_commits_url}/{str(sha)}'
        response, headers = await self.http_client.get(get_commit_info_url)

        file_paths = set()
        for file in response['files']:
            file_paths.add(file['filename'])

        return response, file_paths

    async def batch_get_commit_info(self, batch_size=200):
        self.logger.info(f'Getting batch commit info for {self.github_repo_username_title}')

        commit_info_list = []
        file_paths_set = set()
        all_paths = set()
        written_paths = set() # to avoid duplicate writes

        list_shas = await self.commit_repo.get_all_shas(full=False)
        full_info_shas = await self.commit_repo.get_all_shas(full=True)
        missing_shas = list_shas - full_info_shas

        self.logger.info('Getting commit info for {} commits'.format(len(missing_shas)))

        for sha in missing_shas:
            commit_info, paths = await self.get_commit_info(sha)
            commit_info_list.append(commit_info)

            # filter out directory-like names (require a dot)
            file_paths = {p for p in paths if '.' in p.rsplit('/', 1)[-1]}
            file_paths_set.update(file_paths)
            all_paths.update(file_paths)

            if len(commit_info_list) >= batch_size:
                await self.commit_repo.insert_new_commits(commit_info_list, full=True)
                commit_info_list.clear()

                new_to_write = file_paths_set - written_paths
                if new_to_write:
                    await AsyncDatabase.insert_many(self.file_tracking_collection,
                                                [{'path': file_path} for file_path in new_to_write], data_type='files')
                    written_paths.update(new_to_write)
                file_paths_set.clear()

        if commit_info_list:
            await self.commit_repo.insert_new_commits(commit_info_list, full=True)
        if file_paths_set:
            new_to_write = file_paths_set - written_paths
            if new_to_write:
                await AsyncDatabase.insert_many(self.file_tracking_collection,
                                            [{'path': file_path} for file_path in new_to_write], data_type='files')

        return [{'path': p} for p in all_paths]

    async def get_file_size_at_commit(self, file_path, sha):
        url = f'{self.get_contents_url}/{file_path}?ref={sha}'

        try:
            response, _ = await self.http_client.get(url)

            if isinstance(response, list):
                self.logger.error("Found directory")
                return 'is_directory'
            if response.get('type') == 'symlink':
                return 'is_symlink'
            if isinstance(response, dict) and 'size' in response:
                return response['size']

            self.logger.error(f"Unexpected response structure for {url}: {response}")
            return 'unexpected_response'

        except ClientResponseError as e:
            if e.status == 404:
                return 'file_not_found'
            else:
                raise

    async def get_commit_details(self, sha):
        commit_details = await self.commit_repo.find_commit(sha, full=True)
        return commit_details

    async def is_file_deleted_in_commit(self, commit_details, file_path):
        for commit_detail in commit_details:
            if 'files' in commit_detail:
                for file in commit_detail['files']:
                    if file['filename'] == file_path and file['status'] == 'removed':
                        return True
        return False

    async def get_file_commit_history(self, file_path, update):
        url = f'{self.get_commits_url}?path={file_path}&per_page={APIConnectionAsync.RESULTS_PER_PAGE}'
        full_commit_history = []
        existing_shas = set()

        if update:
            existing_history = await AsyncDatabase.find_one(self.file_tracking_collection, {'path': file_path})
            existing_shas = {commit['sha'] for commit in
                             existing_history.get('commit_history', [])} if existing_history else set()

        while url:
            commits, headers = await self.http_client.get(url)

            if not commits:
                self.logger.warning('No commits found for file {}'.format(file_path))
                break

            for commit in commits:
                sha = commit['sha']

                if update and sha in existing_shas:  # Skip already stored commits in update mode
                    continue

                committer = (
                        (commit.get("committer") or {}).get("login")
                        or (commit.get("commit") or {}).get("committer", {}).get("name")
                        or "unknown"
                )
                commit_date = commit['commit']['author']['date']
                size_or_status = await self.get_file_size_at_commit(file_path, sha)

                if size_or_status in ('unexpected_response', 'is_directory', 'is_symlink'):
                    self.logger.warning(f"Skipping commit {sha} for file {file_path} due to: {size_or_status}")
                    continue

                if size_or_status == 'file_not_found':
                    commit_details = await self.get_commit_details(commit['sha'])
                    renamed = False

                    for file in commit_details.get('files', []):
                        if file['filename'] != file_path:
                            continue

                        if file['status'] == 'removed':
                            self.logger.info(f'File {file_path} deleted in commit {sha}')
                            size_or_status = 0
                            break

                        elif file['status'] == 'renamed' and 'previous_filename' in file:
                            old_file_path = file['previous_filename']
                            self.logger.info(f"File {file_path} was renamed from {old_file_path} in commit {sha}")

                            old_history = await self.get_file_commit_history(old_file_path, update=False)
                            combined = []
                            seen = set()
                            for entry in old_history + full_commit_history:
                                if entry['sha'] not in seen:
                                    seen.add(entry['sha'])
                                    combined.append(entry)

                            await AsyncDatabase.update_one(
                                self.file_tracking_collection,
                                {'path': old_file_path},
                                {
                                    '$set': {
                                        'path': file_path,
                                        'commit_history': combined
                                    },
                                    '$push': {
                                        'previous_paths': old_file_path
                                    }
                                },
                                upsert = True
                            )

                            full_commit_history = combined
                            size_or_status = combined[-1]['size']
                            renamed = True
                            break
                    
                    if renamed:
                        continue

                full_commit_history.append({
                    'sha': sha,
                    'date': commit_date,
                    'size': size_or_status,
                    'committer': committer
                })

            # Get the next page URL from headers if available
            next_url = await self.http_client.get_next_link(headers)

            if not next_url:
                break

            if next_url == url:
                self.logger.error(f"Pagination loop detected for file {file_path}")
                break

            url = next_url

        unique_history = {entry['sha']: entry for entry in full_commit_history}.values()

        if unique_history:
            if update and existing_shas:  # Append to existing history in update mode
                await AsyncDatabase.update_one(
                    self.file_tracking_collection,
                    {'path': file_path},
                    {'$push': {'commit_history': {'$each': list(unique_history)}}}
                )

                self.logger.info(f"Commit history for {file_path} updated with {len(unique_history)} new commits")
            else:
                await AsyncDatabase.insert_many(
                    self.file_tracking_collection,
                    [{'path': file_path, 'commit_history': list(unique_history)}], data_type='files'
                )

        return unique_history

    async def iterate_over_file_paths(self, affected_files, update=False):
        self.logger.info(f'Iterating over the files of {self.github_repo_username_title} to get commit history')

        if update:
            if affected_files:
                files_to_iterate = affected_files
            else:
                self.logger.info("No files to process.")
                return
        else:
            files_to_iterate = await AsyncDatabase.fetch_all(self.file_tracking_collection)

        self.logger.info('Iterating over {} files'.format(len(files_to_iterate)))

        for file in files_to_iterate:
            await self.get_file_commit_history(file['path'], update=update)

        self.logger.info('All files were iterated over')

    async def populate_db(self):
        try:
            commit_list_exists = await self.commit_repo.find_any(full=False)
            update = commit_list_exists is not None

            await self.get_commit_list(update=update)
            file_paths = await self.batch_get_commit_info()
            await self.iterate_over_file_paths(file_paths, update=update)

        except Exception as e:
            self.logger.error(f"An error occurred in populate_db: {e}", exc_info=True)