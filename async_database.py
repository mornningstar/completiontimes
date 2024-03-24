import pandas as pd
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
    async def fetch_all(collection):
        data = await AsyncDatabase.find(collection, query={})
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
    async def find_one(collection, query):
        return await AsyncDatabase.DATABASE[collection].find_one(query)

    @staticmethod
    async def save_dataframe(collection, repository_name, dataframe):
        json_dataframe = dataframe.to_json()
        document = {
            "repository_name": repository_name,
            "dataframe": json_dataframe
        }

        await AsyncDatabase.DATABASE[collection].replace_one(
            {"repository_name": repository_name},
            document,
            upsert=True)

    @staticmethod
    async def load_dataframe(collection, repository_name):
        document = await AsyncDatabase.DATABASE[collection].find_one({"repository_name": repository_name})

        if document:
            dataframe = pd.read_json(document['dataframe'])
            return dataframe

        return None
