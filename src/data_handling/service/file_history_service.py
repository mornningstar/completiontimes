import asyncio
import logging

from aiohttp import ClientResponseError

from src.data_handling.database.file_repo import FileRepository
from src.data_handling.service.commit_sync_service import CommitSyncService
from src.github.http_client import GitHubClient


class FileHistoryService:
    RESULTS_PER_PAGE = CommitSyncService.RESULTS_PER_PAGE

    def __init__(self, github_client: GitHubClient, repo: str,
                 file_repo: FileRepository, commit_service: CommitSyncService):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.http_client = github_client
        self.repo = repo

        self.file_repo = file_repo
        self.base_url = f'https://api.github.com/repos/{self.repo}'
        self.commit_service = commit_service

        self.visited_paths: set[str] = set()
        self.failed_requests = set() # cache requests for files with SHA that yield a 404

    async def build_file_history(self, file_path: str, update: bool = False):
        full_history = []
        url = f"{self.base_url}/commits?path={file_path}&per_page={self.RESULTS_PER_PAGE}"

        known_shas = set()
        if update:
            doc = await self.file_repo.find_file_data(file_path)
            if doc:
                known_shas = {c["sha"] for c in doc.get("commit_history", [])}
        
        if file_path in self.visited_paths:
            self.logger.debug(f"Already visited {file_path}, skipping.")
            doc = await self.file_repo.find_file_data(file_path)
            return doc.get("commit_history", [])

        while url:
            new_entries_in_page = 0
            commits, headers = await self.http_client.get(url)

            if not commits:
                self.logger.warning(f"No commits found for file {file_path}")
                break

            for commit in commits:
                sha = commit['sha']

                if update and sha in known_shas:
                    self.logger.debug(f"Skipping already stored commit {sha} for {file_path}")
                    continue

                new_entries_in_page += 1

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
                        self.logger.debug(f"{file_path} was deleted at commit {sha}")
                        size_or_status = 0

                if size_or_status in ('unexpected_response', 'is_directory', 'is_symlink'):
                    self.logger.warning(f"Skipping commit {sha} for file {file_path} due to: {size_or_status}")
                    continue

                commit_detail = await self.commit_service.commit_repo.find_commit(sha, full=True)
                stats = next(
                    (f for f in commit_detail.get("files", [])
                     if f["filename"] == file_path),
                    None
                )

                additions = stats.get("additions", 0) if stats else 0
                deletions = stats.get("deletions", 0) if stats else 0
                changes = stats.get("changes", 0) if stats else 0

                entry = {
                    'sha': sha,
                    'date': commit['commit']['author']['date'],
                    'committer': (commit.get('committer') or {}).get('login')
                                 or commit['commit']['committer']['name'],
                    'size': size_or_status,
                    'additions': additions,
                    'deletions': deletions,
                    'total_changes': changes
                }

                full_history.append(entry)
            
            if update and new_entries_in_page == 0:
                self.logger.info(f"No new entries for {file_path} found on a full page. Stopping early...")
                break

            next_url = self.http_client.get_next_link(headers)
            url = next_url
        
        self.visited_paths.add(file_path)

        return full_history

    async def get_size_or_status(self, file_path: str, sha: str):
        cache_key = (file_path, sha)
        if cache_key in self.failed_requests:
            return 'file_not_found'

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
                self.failed_requests.add(cache_key)
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

        existing_paths = await self.file_repo.find_existing_paths(unique_paths)
        new_paths = unique_paths - set(existing_paths)

        # bulk-insert if they’re not already there
        await self.file_repo.insert_new_files(
            [{"path": p} for p in new_paths]
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

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                print(f"Worker failed with: {result}")

        self.logger.info('All file histories synced.')
