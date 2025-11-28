import logging

import pymongo
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError

from src.data_handling.database.exceptions import DatabaseError


class AsyncDatabase:
    URI = None
    DATABASE = None
    DATABASE_NAME = None

    @staticmethod
    async def initialize():
        client = AsyncIOMotorClient(AsyncDatabase.URI)
        AsyncDatabase.DATABASE = client[AsyncDatabase.DATABASE_NAME]

    @staticmethod
    async def fetch_all(collection, query=None, projection=None):
        if query is None:
            query = {}

        data = await AsyncDatabase.find(collection, query=query, projection=projection)
        return data

    @staticmethod
    async def fetch_all_shas(collection):
        commits = await AsyncDatabase.find(collection, query={}, projection={'sha': 1})  # Fetch only the SHA field
        return {commit['sha'] for commit in commits}

    @staticmethod
    async def insert(collection, data):
        await AsyncDatabase.DATABASE[collection].insert_one(data)

    @staticmethod
    async def insert_many(collection, data, data_type='commit'):
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
            await AsyncDatabase.DATABASE[collection].bulk_write(operations)

    @staticmethod
    async def find(collection, query, projection=None):
        cursor = AsyncDatabase.DATABASE[collection].find(query, projection)
        return await cursor.to_list(length=None)

    @staticmethod
    async def find_one(collection, query, projection=None):
        return await AsyncDatabase.DATABASE[collection].find_one(query, projection=projection)

    @staticmethod
    async def update_one(collection, filter_query, update_query, upsert=False):
        result = await AsyncDatabase.DATABASE[collection].update_one(filter_query, update_query, upsert=upsert)
        return result

    @staticmethod
    async def delete_one(collection, query):
        try:
            result = await AsyncDatabase.DATABASE[collection].delete_one(query)
            if result.deleted_count == 0:
                logging.warning(f"No document found to delete in collection '{collection}' with query: {query}")
            else:
                logging.info(f"Deleted one document from collection '{collection}' with query: {query}")
            return {"deleted_count": result.deleted_count}
        except PyMongoError as e:
            raise DatabaseError(f"Delete failed in '{collection}' with query: {query}", e)