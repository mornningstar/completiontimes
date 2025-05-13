from src.data_handling.database.async_database import AsyncDatabase


class CommitRepository:
    def __init__(self, repo: str, db = AsyncDatabase):
        self._db = db
        self.collection_name = repo.replace("/", "_")
        self.commit_list_collection = f"{self.collection_name}_commit_list"
        self.full_commit_info_collection = f"{self.collection_name}_full_commit_info"

    # ---------------------------------------------------------------------
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    async def find_any(self, full: bool = False):
        """ Find any commit entry in db collection """
        return await self._db.find_one(self._collection(full), {})

    async def find_commit(self, sha, full: bool = False):
        """ Find a commit by SHA """
        return await self._db.find_one(self._collection(full), {"sha": sha})

    async def insert_new_commits(self, commits, full: bool = False):
        """ Insert an arbitrary amount of commits into db collection """
        return await self._db.insert_many(self._collection(full), commits)

    async def get_all_shas(self, full: bool = False):
        """ Return all available SHAs stored in db collection """
        return await self._db.fetch_all_shas(self._collection(full))

    async def get_all(self, full: bool = False):
        return await self._db.find(self._collection(full), {})

    # --------------------------------------------------------------------- #
    # Helpers                                                               #
    # --------------------------------------------------------------------- #

    def _collection(self, full: bool) -> str:
        """Map the full flag to the correct collection name """
        return self.full_commit_info_collection if full else self.commit_list_collection