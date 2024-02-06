import pandas as pd
from matplotlib import pyplot as plt

from async_database import AsyncDatabase


class FileVisualiser:

    def __init__(self, collection_name, file_path):
        self.collection_name = collection_name
        self.file_path = file_path
        self.file_data = None
        self.size_df = None

    async def fetch_data(self):
        query = {"path": self.file_path}
        self.file_data = await AsyncDatabase.find(self.collection_name, query)

    def process_data(self):
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
