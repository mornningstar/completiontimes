import asyncio
import json
import os

from src.data_handling.database.api_connection_async import APIConnectionAsync
from src.data_handling.database.async_database import AsyncDatabase

REPO = "openedx/edx-platform"

async def retry_failed_commits():
    if not os.path.exists("failed_commits.json"):
        print("No failed_commits.json found.")
        return

    with open("failed_commits.json", "r") as f:
        failed_shas = json.load(f)

    conn = await APIConnectionAsync.create(REPO)

    try:
        async def fetch_and_store(sha: str):
            try:
                info, _ = await conn.get_commit_info(sha)
                await conn.commit_repo.insert_new_commits([info], full=True)
            except Exception as exc:
                print(f"Retry failed for {sha}: {exc}")

        await asyncio.gather(*(fetch_and_store(s) for s in failed_shas))
    finally:
        await conn.close_session()

async def retry_failed_files():
    if not os.path.exists("failed_files.json"):
        print("No failed_files.json found.")
        return

    with open("failed_files.json", "r") as f:
        failed_paths = json.load(f)

    conn = await APIConnectionAsync.create(REPO)

    try:
        async def fetch_and_store(path):
            try:
                await conn.get_file_commit_history(path, update=True)
            except Exception as e:
                print(f"Retry failed for {path}: {e}")

        await asyncio.gather(*(fetch_and_store(path) for path in failed_paths))

    finally:
        await conn.close_session()

async def find_and_refetch_missing_commit_infos():
    conn = await APIConnectionAsync.create(REPO)
    all_shas = await conn.commit_repo.get_all_shas(full=False)
    full_shas = await conn.commit_repo.get_all_shas(full=True)
    missing_shas = all_shas - full_shas

    print(f"Found {len(missing_shas)} missing commit infos.")

    try:
        async def find(sha):
            try:
                commit_info, file_paths = await conn.get_commit_info(sha)
                await conn.commit_repo.insert_new_commits([commit_info], full=True)
            except Exception as e:
                print(f"Failed to refetch {sha}: {e}")

        await asyncio.gather(*(find(sha) for sha in missing_shas))
    finally:
        await conn.close_session()

async def find_files_with_empty_or_missing_history():
    conn = await APIConnectionAsync.create(REPO)

    files_with_issues = []
    all_files = await AsyncDatabase.fetch_all(conn.file_tracking_collection)

    for file in all_files:
        if not file.get("commit_history"):
            files_with_issues.append(file["path"])

    print(f"{len(files_with_issues)} files with empty or missing commit_history.")

    with open("incomplete_file_histories.json", "w") as f:
        json.dump(files_with_issues, f, indent=2)

async def retry_incomplete_file_histories():
    if not os.path.exists("incomplete_file_histories.json"):
        print("No incomplete_file_histories.json found.")
        return

    with open("incomplete_file_histories.json", "r") as f:
        incomplete_paths = json.load(f)

    conn = await APIConnectionAsync.create(REPO)

    try:
        async def _fix(path: str) -> None:
            try:
                await conn.get_file_commit_history(path, update=True)
            except Exception as exc:
                print(f"Retry failed for incomplete file {path}: {exc}")

        await asyncio.gather(*(_fix(p) for p in incomplete_paths))
    finally:
            await conn.close_session()

if __name__ == "__main__":
    async def main():

        await find_and_refetch_missing_commit_infos()
        await find_files_with_empty_or_missing_history()
        await retry_failed_commits()
        await retry_failed_files()
        await retry_incomplete_file_histories()

    asyncio.run(main())