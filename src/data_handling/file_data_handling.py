import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_handling.async_database import AsyncDatabase

def add_features(dataframe, size_col='size', window=7):
    # Rolling mean of file size over the window
    dataframe[f'rolling_{window}_size'] = dataframe[size_col].rolling(window=window).mean()

    # Rolling standard deviation of file size
    dataframe[f'rolling_{window}_std'] = dataframe[size_col].rolling(window=window).std()

    # Exponential moving average (EMA) of file size
    dataframe[f'size_ema'] = dataframe[size_col].ewm(span=window, adjust=False).mean()

    # Cumulative file size
    dataframe['cumulative_size'] = dataframe[size_col].cumsum()

    # Lag features
    dataframe[f'lag_{window}_size'] = dataframe[size_col].shift(window)

    return dataframe

class FileDataHandler:
    def __init__(self, collection_name, file_path):
        self.collection_name = collection_name
        self.file_path = file_path
        self.file_data = None
        self.size_df = None

    async def fetch_data(self):
        query = {"path": self.file_path}
        self.file_data = await AsyncDatabase.find(self.collection_name, query)
        print(f"File data: {len(self.file_data)}")
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
        })

        df.set_index('time', inplace=True)
        df.index = pd.DatetimeIndex(df.index)
        df.sort_values(by='time', inplace=True)
        self.size_df = df.resample('D').ffill()

        self.size_df = add_features(self.size_df)

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

        print("I landed here")

        self.size_df.sort_index(inplace=True)

        self.size_df.dropna(subset=['size'], inplace=True)

        train_size = 1 - test_size
        train, test = train_test_split(self.size_df, train_size=train_size, shuffle=False)

        x_train = train.index.astype(int).values.reshape(-1, 1)
        y_train = train['size']

        x_test = test.index.astype(int).values.reshape(-1, 1)
        y_test = test['size']

        return x_train, y_train, x_test, y_test

    def prepare_lstm_data(self, timesteps=10, test_size=0.2):
        """
        Prepares data for LSTM models (3D input for LSTM).
        :param timesteps: Number of time steps to consider in the LSTM input
        :param test_size: Percentage of data to be used for testing
        :return:
        x_train, y_train: 3D inputs for LSTM training,
        x_test, y_test: 3D inputs for LSTM testing
        """

        print("I landed correctly")

        print(self.size_df.head())

        self.size_df.sort_index(inplace=True)
        self.size_df.ffill(inplace=True)
        self.size_df.bfill(inplace=True)

        # Split into training and test sets
        train_size = int((1 - test_size) * len(self.size_df))
        df_train = self.size_df[:train_size]
        df_test = self.size_df[train_size:]

        print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
        print(f"Timesteps: {timesteps}")

        # Prepare LSTM-specific 3D data [samples, timesteps, features]
        x_train = np.array([df_train.values[i:i + timesteps] for i in range(len(df_train) - timesteps)])
        y_train = df_train['size'].values[timesteps:]

        x_test = np.array([df_test.values[i:i + timesteps] for i in range(len(df_test) - timesteps)])
        y_test = df_test['size'].values[timesteps:]

        print(f"x_train shape: {x_train.shape}")  # Should be (samples, timesteps, features)
        print(f"y_train shape: {y_train.shape}")  # Should be (samples,)
        print(f"x_test shape: {x_test.shape}")  # Should be (samples, timesteps, features)
        print(f"y_test shape: {y_test.shape}")  # Should be (samples,)

        return x_train, y_train, x_test, y_test

