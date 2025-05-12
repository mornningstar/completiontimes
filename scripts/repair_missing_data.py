import asyncio
import json
import os

from src.data_handling.database.api_connection_async import APIConnectionAsync

REPO = "openedx/edx-platform"

async def retry_failed_commits():
    if not os.path.exists("failed_commits.json"):
        print("No failed_commits.json found.")
        return

    with open("failed_commits.json", "r") as f:
        failed_shas = json.load(f)

    conn = await APIConnectionAsync.create(REPO)
    for sha in failed_shas:
        try:
            await conn.get_commit_info(sha)
        except Exception as e:
            print(f"Retry failed for {sha}: {e}")

    await conn.close_session()

async def retry_failed_files():
    if not os.path.exists("failed_files.json"):
        print("No failed_files.json found.")
        return

    with open("failed_files.json", "r") as f:
        failed_paths = json.load(f)

    conn = await APIConnectionAsync.create(REPO)

    for path in failed_paths:
        try:
            await conn.get_file_commit_history(path, update=True)
        except Exception as e:
            print(f"Retry failed for {path}: {e}")

    await conn.close_session()

async def find_and_refetch_missing_commit_infos():
    conn = await APIConnectionAsync.create(REPO)
    all_shas = await conn.repo.fetch_all_commit_shas()
    full_shas = await conn.repo.fetch_all_full_commit_shas()
    missing_shas = all_shas - full_shas

    print(f"Found {len(missing_shas)} missing commit infos.")

    for sha in missing_shas:
        try:
            commit_info, file_paths = await conn.get_commit_info(sha)
            await conn.repo.insert_full_commit_info([commit_info])
        except Exception as e:
            print(f"Failed to refetch {sha}: {e}")

    await conn.close_session()

async def find_files_with_empty_or_missing_history():
    conn = await APIConnectionAsync.create(REPO)

    files_with_issues = []
    all_files = await conn.repo.fetch_all_file_tracking()

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

    for path in incomplete_paths:
        try:
            await conn.get_file_commit_history(path, update=True)
        except Exception as e:
            print(f"Retry failed for incomplete file {path}: {e}")

    await conn.close_session()

if __name__ == "__main__":
    async def main():

        await find_and_refetch_missing_commit_infos()
        await find_files_with_empty_or_missing_history()
        await retry_failed_commits()
        await retry_failed_files()
        await retry_incomplete_file_histories()

    asyncio.run(main())