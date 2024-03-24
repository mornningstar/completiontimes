import pandas as pd
from matplotlib import pyplot as plt

from async_database import AsyncDatabase


class FileVisualiser:
    DATAFRAME_COLLECTION = "dataframes_files"

    def __init__(self, collection_name, file_path):
        self.data_df = None
        self.collection_name = collection_name
        self.file_path = file_path
        self.file_data = None
        self.size_df = None

    @property
    def dataframe_identifier(self):
        return f"{self.collection_name}_{self.file_path.replace('/', '_')}"

    async def fetch_data(self):
        self.data_df = await AsyncDatabase.load_dataframe(self.DATAFRAME_COLLECTION, self.dataframe_identifier)

        if self.data_df is not None:
            await self.process_data(self.data_df)
        else:
            query = {"path": self.file_path}
            self.file_data = await AsyncDatabase.find(self.collection_name, query)
            await self.process_data()

    async def process_data(self, df=None):
        if df is None:
            times = []
            sizes = []

            for record in self.file_data:
                for commit in record.get('commit_history', []):
                    commit_date = pd.to_datetime(commit['date'])
                    file_size = commit.get('size', 0)

                    times.append(commit_date)
                    sizes.append(file_size)

            df = pd.DataFrame({
                'time': times,
                'size': sizes
            })

            df.sort_values(by='time', inplace=True)
            df.set_index('time', inplace=True)

            await AsyncDatabase.save_dataframe(self.DATAFRAME_COLLECTION, self.collection_name, df)

        self.size_df = df

    def plot_data(self):
        if self.size_df is None:
            raise ValueError('Data is not processed. Call process_data() before plotting!')

        plt.figure(figsize=(12, 6))
        plt.plot(self.size_df.index, self.size_df['size'], label='File Size')

        plt.xlabel('Date')
        plt.ylabel('Size')
        plt.title(f'File Size Over Time for {self.file_path}')
        plt.grid(True)
        plt.legend()

        plt.savefig(f'images/plot_{self.file_path.replace("/", "_")}.png')
