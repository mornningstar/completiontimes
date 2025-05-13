import logging

from aiohttp import ClientResponseError

from src.data_handling.database.file_repo import FileRepository
from src.data_handling.service.commit_sync_service import CommitSyncService
from src.github.http_client import GitHubClient


class FileHistoryService:
    RESULTS_PER_PAGE = CommitSyncService.RESULTS_PER_PAGE

    def __init__(self, github_client: GitHubClient, repo: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.http_client = github_client
        self.repo = repo

        self.file_repo = FileRepository(self.repo)
        self.base_url = f'https://api.github.com/repos/{self.repo}'
        self.commit_service = CommitSyncService(github_client, repo)
    
    async def build_file_history(self, file_path: str, update: bool = False):
        full_history = []
        written_shas = set()

        if update:
            existing = await self.file_repo.find_file_data(file_path)
            if existing and existing.get('commit_history'):
                for entry in existing['commit_history']:
                    full_history.append(entry)
                    written_shas.add(entry['sha'])

        url = f"{self.base_url}/commits?path={file_path}&per_page={self.RESULTS_PER_PAGE}"

        while url:
            commits, headers = await self.http_client.get(url)

            if not commits:
                self.logger.warning(f"No commits found for file {file_path}")
                break

            for commit in commits:
                sha = commit['sha']

                if sha in written_shas:
                    continue

                size_or_status = await self.get_size_or_status(file_path, sha)

                if size_or_status == 'file_not_found':
                    renamed, new_path, renamed_history = await self._handle_rename(file_path, sha)

                    if renamed:
                        file_path = new_path
                        for entry in renamed_history:
                            if entry['sha'] not in written_shas:
                                full_history.append(entry)
                                written_shas.add(entry['sha'])
                                
                        continue
                    else:
                        # File deleted: record size 0
                        size_or_status = 0
                
                if size_or_status in ('unexpected_response', 'is_directory', 'is_symlink'):
                    self.logger.warning(f"Skipping commit {sha} for file {file_path} due to: {size_or_status}")
                    continue

                entry = {
                    'sha': sha,
                    'date': commit['commit']['author']['date'],
                    'committer': (commit.get('committer') or {}).get('login')
                                 or commit['commit']['committer']['name'],
                    'size': size_or_status
                }
                full_history.append(entry)
                written_shas.add(sha)

            next_url = await self.http_client.get_next_link(headers)
            if not next_url:
                break
            url = next_url

        if update:
            if full_history:
                await self.file_repo.update_file_data(file_path, file_path, full_history, upsert=True)
                self.logger.info(f"Updated history for {file_path} with {len(full_history)} entries")
        else:
            if full_history:
                await self.file_repo.insert_file_with_history(file_path, full_history)
                self.logger.info(f"Inserted history for {file_path} with {len(full_history)} entries")

        return full_history

    async def get_size_or_status(self, file_path: str, sha: str):
        try:
            url = f"{self.base_url}/contents/{file_path}?ref={sha}"
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
            raise

    async def _handle_rename(self, file_path: str, sha: str):
        """Helper to detect rename/deletion in a missing-file commit"""
        commit_detail = await self.commit_service.commit_repo.find_commit(sha, full=True)
        for file in commit_detail.get('files', []):
            if file['status'] == 'renamed' and file.get('previous_filename'):
                old_path = file['previous_filename']
                new_path = file['filename']
                # Recursively pull history for old path
                old_hist = await self.build_file_history(old_path, update=False)
                return True, new_path, old_hist
            if file['status'] == 'removed' and file['filename'] == file_path:
                return False, file_path, []
        return False, file_path, []

    async def collect_all_paths_from_commits(self) -> list[str]:
        """
        Scan CommitRepository (full=True) once and store every unique path
        into FileRepository. Returns the list of paths.
        """
        commits = await self.commit_service.commit_repo.find_all(full=True)

        unique_paths = set()
        for commit in commits:
            for file in commit.get("files", []):
                if "." not in file["filename"].rsplit("/", 1)[-1]:
                    continue
                unique_paths.add(file["filename"])

        # bulk-insert if they’re not already there
        await self.file_repo.insert_new_files(
            [{"path": p} for p in unique_paths if not await self.file_repo.find_file_data(p)]
        )
        return list(unique_paths)

    async def sync_all_file_histories(self, update: bool = False):
        paths = await self.file_repo.get_all()
        for file in paths:
            await self.build_file_history(file['path'], update=update)

        self.logger.info('All file histories synced.')
