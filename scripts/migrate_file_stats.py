import asyncio

import pandas as pd

from src.data_handling.database.async_database import AsyncDatabase
from src.data_handling.database.commit_repo import CommitRepository
from src.data_handling.database.file_repo import FileRepository


async def gather_commit_stats(repo_name: str):
    await AsyncDatabase.initialize()
    file_repo = FileRepository(repo_name)
    commit_repo = CommitRepository(repo_name)

    all_files = await file_repo.get_all()

    for file_data in all_files:
        commit_history = []
        path = file_data['path']
        for commit in file_data.get('commit_history', []):
            commit_data = await commit_repo.find_commit(commit['sha'], full=True)
            commit_files = commit_data['files']
            for file in commit_files:
                if file['filename'] == path:
                    commit_history.append({
                        'sha': commit.get('sha'),
                        'date': commit.get('date'),
                        'additions': file.get('additions'),
                        'deletions': file.get('deletions'),
                        'changes': file.get('changes'),
                    })
                    break

        await file_repo.replace_commit_history(path, commit_history)


def main():
    repo = "khoj-ai/khoj"
    asyncio.run(gather_commit_stats(repo))


if __name__ == "__main__":
    main()