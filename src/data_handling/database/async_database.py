import logging

import pymongo
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError

from src.data_handling.database.exceptions import DatabaseError


class AsyncDatabase:
    def __init__(self, uri: str, database_name: str):
        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[database_name]
        self.logger = logging.getLogger(__name__)

    async def fetch_all(self, collection, query=None, projection=None):
        try:
            if query is None:
                query = {}

            return await self.find(collection, query=query, projection=projection)
        except PyMongoError as e:
            raise DatabaseError(f"Fetch all failed in '{collection}'", e)

    async def fetch_all_shas(self, collection):
        try:
            commits = await self.find(collection, query={}, projection={'sha': 1})
            return {commit['sha'] for commit in commits}
        except PyMongoError as e:
            raise DatabaseError(f"Fetch SHAs failed in '{collection}'", e)

    async def insert(self, collection, data):
        try:
            await self.db[collection].insert_one(data)
        except PyMongoError as e:
            raise DatabaseError(f"Insert failed in '{collection}'", e)

    async def insert_many(self, collection, data, data_type='commit'):
        try:
            operations = []

            if data_type == 'commit':
                for commit in data:
                    operation = pymongo.UpdateOne(
                        {'sha': commit['sha']},
                        {'$setOnInsert': commit},
                        upsert=True
                    )
                    operations.append(operation)

            elif data_type == 'files':
                for datapoint in data:
                    operation = pymongo.UpdateOne(
                        {'path': datapoint['path']},
                        {'$set': datapoint},
                        upsert=True
                    )
                    operations.append(operation)

            if operations:
                await self.db[collection].bulk_write(operations)
        except PyMongoError as e:
            raise DatabaseError(f"Bulk insert failed in '{collection}'", e)

    async def find(self, collection, query, projection=None):
        try:
            cursor = self.db[collection].find(query, projection)
            return await cursor.to_list(length=None)
        except PyMongoError as e:
            raise DatabaseError(f"Find failed in '{collection}'", e)

    async def find_one(self, collection, query, projection=None):
        try:
            return await self.db[collection].find_one(query, projection=projection)
        except PyMongoError as e:
            raise DatabaseError(f"Find one failed in '{collection}'", e)

    async def update_one(self, collection, filter_query, update_query, upsert=False):
        try:
            return await self.db[collection].update_one(filter_query, update_query, upsert=upsert)
        except PyMongoError as e:
            raise DatabaseError(f"Update failed in '{collection}'", e)

    async def delete_one(self, collection, query):
        try:
            result = await self.db[collection].delete_one(query)
            if result.deleted_count == 0:
                self.logger.warning(f"No document found to delete in collection '{collection}' with query: {query}")
            else:
                self.logger.info(f"Deleted one document from collection '{collection}' with query: {query}")
            return {"deleted_count": result.deleted_count}
        except PyMongoError as e:
            raise DatabaseError(f"Delete failed in '{collection}' with query: {query}", e)