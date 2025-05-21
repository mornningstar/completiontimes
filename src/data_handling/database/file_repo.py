from src.data_handling.database.async_database import AsyncDatabase


class FileRepository:
    """ Class for all Mongo CRUD operations around files. """
    def __init__(self, repo: str, db = AsyncDatabase):
        self._db = db
        self.collection_name = repo.replace("/", "_")
        self.file_tracking_collection = f"{self.collection_name}_file_tracking"

    # ---------------------------------------------------------------------
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    async def get_all(self):
        return await self._db.fetch_all(self.file_tracking_collection)

    async def insert_new_files(self, file_paths: list):
        return await self._db.insert_many(self.file_tracking_collection, file_paths, data_type='files')

    async def find_file_data(self, file_path: str):
        return await self._db.find_one(self.file_tracking_collection, {'path': file_path})

    async def has_commit_for_file(self, file_path: str, sha: str):
        doc = await self._db.find_one(self.file_tracking_collection,
                                      {"path": file_path, "commit_history.sha": sha}, projection={"_id": 1}
                                      )
        return doc is not None

    async def update_file_data(self, old_path: str, new_path: str, combined_history: list[str], upsert: bool = True):
        query = {'path': old_path}
        update = {
            '$set': {
                'path': new_path,
                'commit_history': combined_history
            },
            '$push': {
                'previous_paths': old_path
            }
        }

        return await self._db.update_one(self.file_tracking_collection, query, update, upsert=upsert)

    async def append_commit_history(self, file_path: str, new_commits: list, upsert: bool = False):
        query = {'path': file_path}
        update = {'$push':
                      {'commit_history':
                           {'$each': new_commits
                            }
                       }
                  }

        return await self._db.update_one(self.file_tracking_collection, query, update, upsert=upsert)

    async def insert_file_with_history(self, file_path: str, commit_history: list):
        doc = [{'path': file_path, 'commit_history': commit_history}]

        return await self._db.insert_many(self.file_tracking_collection, doc, data_type='files')

    async def replace_commit_history(self, file_path: str, new_history: list[dict]):
        return await self._db.update_one(self.file_tracking_collection, {"path": file_path},
            {"$set": {"commit_history": new_history}}
        )

    async def append_features_to_file(self, path, features, upsert: bool = False):
        query = {"path": path}
        update = {"$set": {"features": features}}
        return await self._db.update_one(self.file_tracking_collection, query, update)

    async def delete_file_data(self, path):
        query = {"path": path}
        return await self._db.delete_one(self.file_tracking_collection, query)

    async def find_existing_paths(self, paths: set[str]) -> list[str]:
        results = await self._db.find(self.file_tracking_collection,
                                      {"path": {"$in": list(paths)}},
                                      {"path": 1})
        return [result["path"] for result in results]