import logging

from src.data_handling.service.commit_sync_service import CommitSyncService
from src.data_handling.service.file_history_service import FileHistoryService
from src.github.http_client import GitHubClient


class SyncOrchestrator:
    def __init__(self, repo: str,
                 http_client: GitHubClient, commit_service: CommitSyncService, file_service: FileHistoryService):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.http_client = http_client
        self.repo = repo
        self.commit_service = commit_service
        self.file_service = file_service

    async def run(self):
        await self.http_client.open()
        result = await self.commit_service.commit_repo.find_any(full=False)
        update = result is not None

        try:
            self.logger.info("Running commit-file syncing...")
            # ---- commit side ----
            await self.commit_service.sync_commit_list(update)
            await self.commit_service.sync_commit_details()

            # ---- file side ----
            await self.file_service.collect_all_paths_from_commits()
            await self.file_service.sync_all_file_histories(update)

            self.logger.info("Finished full commit-file syncing!")
        except Exception as e:
            self.logger.error('Sync failed', exc_info=True)
        finally:
            await self.http_client.close()
