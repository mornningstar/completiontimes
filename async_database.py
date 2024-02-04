import pymongo
from motor.motor_asyncio import AsyncIOMotorClient


class AsyncDatabase:
    URI = "mongodb://localhost:27017"
    DATABASE = None

    @staticmethod
    async def initialize():
        client = AsyncIOMotorClient(AsyncDatabase.URI)
        AsyncDatabase.DATABASE = client['github_data']

    @staticmethod
    async def fetch_all_shas(collection):
        commits = await AsyncDatabase.find(collection, query={}, projection={'sha': 1})  # Fetch only the SHA field
        return {commit['sha'] for commit in commits}

    @staticmethod
    async def insert(collection, data):
        await AsyncDatabase.DATABASE[collection].insert_one(data)

    @staticmethod
    async def insert_many(collection, data):
        operations = []

        for commit in data:
            operation = pymongo.UpdateOne(
                {'sha': commit['sha']},
                {'$setOnInsert': commit},
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
    async def find_one(collection, query):
        return await AsyncDatabase.DATABASE[collection].find_one(query)
