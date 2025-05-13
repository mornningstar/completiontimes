import asyncio
import logging

from config.config import CONFIG
from src.data_handling.database.async_database import AsyncDatabase
from src.data_handling.database.commit_repo import CommitRepository
from src.data_handling.database.file_repo import FileRepository
from src.github.http_client import GitHubClient


async def migrate_add_committer(file_repo: FileRepository, commit_repo: CommitRepository):
    await AsyncDatabase.initialize()

    files = await file_repo.get_all()

    for file in files:
        path = file["path"]
        updated_history = []

        for commit in file.get("commit_history", []):
            sha = commit.get("sha")
            if not sha:
                continue

            full_info = await commit_repo.find_commit(sha, full=True)
            if not full_info:
                continue

            if full_info.get("committer") and full_info["committer"].get("login"):
                committer = full_info["committer"]["login"]
            else:
                committer = full_info.get("commit", {}).get("committer", {}).get("name", "unknown")

            commit["committer"] = committer
            updated_history.append(commit)

        await file_repo.replace_commit_history(path, updated_history)

        logging.info(f"Updated committer info for file: {path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def main():
        repo = "openedx/edx-platform"
        auth = CONFIG[0]["github_access_token"]
        client = GitHubClient(auth)

        try:
            await client.open()
            await AsyncDatabase.initialize()
            file_repo = FileRepository(repo)
            commit_repo = CommitRepository(repo)

            await migrate_add_committer(file_repo, commit_repo)
        finally:
            await client.close()

    asyncio.run(main())
