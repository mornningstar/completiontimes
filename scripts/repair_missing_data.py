import asyncio
import json
import os

from config.config import CONFIG
from src.data_handling.database.async_database import AsyncDatabase
from src.data_handling.service.commit_sync_service import CommitSyncService
from src.data_handling.service.file_history_service import FileHistoryService
from src.github.http_client import GitHubClient

REPO = "openedx/edx-platform"
AUTH = CONFIG[0]["github_access_token"]

async def retry_failed_commits(commit_service: CommitSyncService):
    if not os.path.exists("failed_commits.json"):
        print("No failed_commits.json found.")
        return

    with open("failed_commits.json", "r") as f:
        failed_shas = json.load(f)

    async def fetch_and_store(sha: str):
        try:
            info, _ = await commit_service.get_commit_info(sha)
            await commit_service.commit_repo.insert_new_commits([info], full=True)
        except Exception as exc:
            print(f"Retry failed for {sha}: {exc}")

    await asyncio.gather(*(fetch_and_store(s) for s in failed_shas))

async def retry_failed_files(file_service: FileHistoryService):
    if not os.path.exists("failed_files.json"):
        print("No failed_files.json found.")
        return

    with open("failed_files.json") as f:
        failed_paths = json.load(f)

    async def fetch_and_store(path: str):
        try:
            await file_service.build_file_history(path, update=True)
        except Exception as e:
            print(f"Retry failed for {path}: {e}")

    await asyncio.gather(*(fetch_and_store(p) for p in failed_paths))

async def find_and_refetch_missing_commit_infos(commit_service: CommitSyncService):
    all_shas = await conn.commit_repo.get_all_shas(full=False)
    full_shas = await conn.commit_repo.get_all_shas(full=True)
    missing_shas = all_shas - full_shas

    print(f"Found {len(missing_shas)} missing commit infos.")

    async def refetch(sha: str):
        try:
            info, _ = await commit_service.get_commit_info(sha)
            await commit_service.commit_repo.insert_new_commits([info], full=True)
        except Exception as e:
            print(f"Failed to refetch {sha}: {e}")

    await asyncio.gather(*(refetch(s) for s in missing_shas))

async def find_files_with_empty_or_missing_history():
    from src.data_handling.database.file_repo import FileRepository
    file_repo = FileRepository(REPO)

    files_with_issues = []
    all_files = await file_repo.get_all()

    for f in all_files:
        if not f.get("commit_history"):
            files_with_issues.append(f["path"])

    print(f"{len(files_with_issues)} files with empty or missing commit_history.")
    with open("incomplete_file_histories.json", "w") as out:
        json.dump(files_with_issues, out, indent=2)

async def retry_incomplete_file_histories(file_service: FileHistoryService):
    if not os.path.exists("incomplete_file_histories.json"):
        print("No incomplete_file_histories.json found.")
        return

    with open("incomplete_file_histories.json") as f:
        incomplete = json.load(f)

    async def fix(path: str):
        try:
            await file_service.build_file_history(path, update=True)
        except Exception as exc:
            print(f"Retry failed for incomplete file {path}: {exc}")

    await asyncio.gather(*(fix(p) for p in incomplete))

if __name__ == "__main__":
    async def main():
        client = GitHubClient(AUTH)
        await client.open()
        await AsyncDatabase.initialize()

        commit_service = CommitSyncService(client, REPO)
        file_service = FileHistoryService(client, REPO)

        try:
            await find_and_refetch_missing_commit_infos(commit_service)
            await find_files_with_empty_or_missing_history()
            await retry_failed_commits(commit_service)
            await retry_failed_files(file_service)
            await retry_incomplete_file_histories(file_service)
        finally:
            await client.close()

    asyncio.run(main())