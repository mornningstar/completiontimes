import asyncio
import logging

from src.data_handling.database.api_connection_async import APIConnectionAsync
from src.data_handling.database.async_database import AsyncDatabase


async def migrate_add_committer(api_connection):
    await AsyncDatabase.initialize()

    files = await AsyncDatabase.fetch_all(api_connection.file_tracking_collection)

    for file in files:
        path = file["path"]
        updated_history = []

        for commit in file.get("commit_history", []):
            sha = commit.get("sha")
            if not sha:
                continue

            full_info = await AsyncDatabase.find_one(api_connection.full_commit_info_collection, {"sha": sha})
            if not full_info:
                continue

            if full_info.get("committer") and full_info["committer"].get("login"):
                committer = full_info["committer"]["login"]
            else:
                committer = full_info.get("commit", {}).get("committer", {}).get("name", "unknown")

            #committer = full_info.get("committer", {}).get("login", "unknown")
            commit["committer"] = committer
            updated_history.append(commit)

        await AsyncDatabase.update_one(
            api_connection.file_tracking_collection,
            {"path": path},
            {"$set": {"commit_history": updated_history}}
        )

        logging.info(f"Updated committer info for file: {path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def main():
        repo = "openedx/edx-platform"
        api_conn = await APIConnectionAsync.create(repo)
        await migrate_add_committer(api_conn)
        await api_conn.close_session()

    asyncio.run(main())
