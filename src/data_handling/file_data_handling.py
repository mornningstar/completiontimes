import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_handling.async_database import AsyncDatabase


class FileDataHandler:
    def __init__(self, collection_name, file_path):
        self.collection_name = collection_name
        self.file_path = file_path
        self.file_data = None
        self.size_df = None

    async def fetch_data(self):
        query = {"path": self.file_path}
        self.file_data = await AsyncDatabase.find(self.collection_name, query)
        self.process_data()

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
        }).set_index('time').sort_values('time')

        df.index = pd.DatetimeIndex(df.index)
        self.size_df = df

    def prepare_data(self, test_size=0.2):
        """
        Prepares the data for visualisation.
        :param test_size: Percentage of data to be used for testing
        :return:
            x_train: timestamps of train datatest,
            x_test: timestamps of test dataset,
            y_train: file sizes of train dataset,
            y_test: file sizes of test dataset
        """
        self.size_df.sort_index(inplace=True)

        train_size = 1 - test_size
        train, test = train_test_split(self.size_df, train_size=train_size, shuffle=False)

        x_train = train.index.astype(int).values.reshape(-1, 1)
        y_train = train['size']

        x_test = test.index.astype(int).values.reshape(-1, 1)
        y_test = test['size']

        return x_train, y_train, x_test, y_test
