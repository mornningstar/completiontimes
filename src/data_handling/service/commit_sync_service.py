import logging

from src.data_handling.database.commit_repo import CommitRepository
from src.github.http_client import GitHubClient


class CommitSyncService:
    RESULTS_PER_PAGE = 100

    def __init__(self, github_client: GitHubClient, repo: str, commit_repo: CommitRepository):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.http_client = github_client
        self.repo = repo

        self.commit_repo = commit_repo
        self.base_url = f'https://api.github.com/repos/{repo}'

    async def sync_commit_list(self, update: bool = False, batch_size: int = 500):
        url = f'{self.base_url}/commits?per_page={self.RESULTS_PER_PAGE}'
        buffer = []
        counter = 0

        async for page, headers in self.http_client.paginate(url):
            if update:
                new_commits = []
                for commit in page:
                    if not await self.commit_repo.find_commit(commit['sha'], full=False):
                        new_commits.append(commit)
                buffer.extend(new_commits)

                if not new_commits:
                    self.logger.info("Encountered full page of already-synced commits — stopping early.")
                    break
            else:
                buffer.extend(page)

            if len(buffer) >= batch_size:
                counter += len(buffer)
                await self.commit_repo.insert_new_commits(buffer, full=False)
                buffer.clear()
        
        if buffer:
            counter += len(buffer)
            await self.commit_repo.insert_new_commits(buffer, full=False)

        self.logger.info(f"Found {counter} new commits...")

    async def get_commit_info(self, sha: str) -> tuple[dict, set[str]]:
        data, _ = await self.http_client.get(f'{self.base_url}/commits/{sha}')
        paths = {f['filename'] for f in data.get('files', [])}
        return data, paths

    async def sync_commit_details(self, batch_size: int = 500):
        list_shas = await self.commit_repo.get_all_shas(full=False)
        full_info_shas = await self.commit_repo.get_all_shas(full=True)
        missing_shas = list_shas - full_info_shas

        if len(missing_shas) == 0:
            self.logger.info("No missing commit details. Skipping.")
            return

        self.logger.info('Getting commit info for {} commits'.format(len(missing_shas)))

        buffer = []
        for sha in missing_shas:
            url = f'{self.base_url}/commits/{sha}'
            data, _ = await self.http_client.get(url)

            buffer.append(data)

            if len(buffer) >= batch_size:
                await self.commit_repo.insert_new_commits(buffer, full=True)
                buffer.clear()

        if buffer:
            await self.commit_repo.insert_new_commits(buffer, full=True)

        self.logger.info('Commit details sync complete.')
