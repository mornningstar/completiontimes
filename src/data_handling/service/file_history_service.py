import asyncio
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

        self.visited_paths: set[str] = set()

    async def build_file_history(self, file_path: str, update: bool = False):
        full_history = []
        url = f"{self.base_url}/commits?path={file_path}&per_page={self.RESULTS_PER_PAGE}"
        skipped = 0
        
        if file_path in self.visited_paths:
            self.logger.debug(f"Already visited {file_path}, skipping.")
            doc = await self.file_repo.find_file_data(file_path)
            return doc.get("commit_history", [])

        while url:
            commits, headers = await self.http_client.get(url)

            if not commits:
                self.logger.warning(f"No commits found for file {file_path}")
                break

            for commit in commits:
                sha = commit['sha']

                if update and await self.file_repo.has_commit_for_file(file_path, sha):
                    self.logger.debug(f"Skipping already stored commit {sha} for {file_path}")
                    continue

                size_or_status = await self.get_size_or_status(file_path, sha)

                if size_or_status == 'file_not_found':
                    renamed, renamed_history = await self._handle_rename(file_path, sha)

                    if renamed:
                        full_history.extend(renamed_history)
                        # TODO: currently discards the rename commit itself
                        self.logger.info(f"Rename detected at {sha} for {file_path} — see commit details below:")
                        commit_detail = await self.commit_service.commit_repo.find_commit(sha, full=True)
                        for f in commit_detail.get("files", []):
                            if f["status"] == "renamed":
                                self.logger.info(f)

                        continue
                    else:
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

            next_url = await self.http_client.get_next_link(headers)
            url = next_url
        
        if skipped > 0:
            self.logger.info(f"{skipped} commits skipped (already stored) for {file_path}")
        
        self.visited_paths.add(file_path)

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
            if file['status'] == 'renamed' and file.get('previous_filename') and file['filename'] == file_path:
                old_path = file['previous_filename']

                # Recursively pull history for old path
                old_hist = await self.build_file_history(old_path, update=False)
                await self.file_repo.delete_file_data(old_path)
                return True, old_hist
            if file['status'] == 'removed' and file['filename'] == file_path:
                return False, []
        return False, []

    async def collect_all_paths_from_commits(self) -> list[str]:
        """
        Scan CommitRepository (full=True) once and store every unique path
        into FileRepository. Returns the list of paths.
        """
        commits = await self.commit_service.commit_repo.get_all(full=True)

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

    async def sync_all_file_histories(self, update: bool = False, max_concurrency: int = 10):
        paths = await self.file_repo.get_all()
        sem = asyncio.BoundedSemaphore(max_concurrency)

        async def worker(path):
            async with sem:
                try:
                    history = await self.build_file_history(path, update=update)
                    if history:
                        if update:
                            await self.file_repo.append_commit_history(path, history, upsert=True)
                        else:
                            await self.file_repo.insert_file_with_history(path, history)
                except Exception as e:
                    self.logger.error(f"Sync failed for {path}: {e!r}")

        tasks = [asyncio.create_task(worker(f['path'])) for f in paths]
        await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info('All file histories synced.')
