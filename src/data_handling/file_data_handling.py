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
        self.filedata_df = None

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
        self.filedata_df = df.resample('D').ffill()

        self.filedata_df = add_features(self.filedata_df)

    def prepare_data(self, target, test_size=0.2):
        """
        Prepares the data for visualisation.
        :param target: target column - which column to predict
        :param test_size: Percentage of data to be used for testing
        :return:
            x_train: timestamps of train datatest,
            x_test: timestamps of test dataset,
            y_train: file sizes of train dataset,
            y_test: file sizes of test dataset
        """
        if target not in self.filedata_df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        self.filedata_df.sort_index(inplace=True)
        self.filedata_df.dropna(subset=[target], inplace=True)

        train, test = train_test_split(self.filedata_df, test_size=test_size, shuffle=False)

        x_train = train.index.astype(int).values.reshape(-1, 1)
        y_train = train[target]

        x_test = test.index.astype(int).values.reshape(-1, 1)
        y_test = test[target]

        return x_train, y_train, x_test, y_test

    def prepare_lstm_data(self, target, timesteps=10, test_size=0.2):
        """
        Prepares data for LSTM models (3D input for LSTM).
        :param target: target column - which column to predict
        :param timesteps: Number of time steps to consider in the LSTM input
        :param test_size: Percentage of data to be used for testing
        :return:
        x_train, y_train: 3D inputs for LSTM training,
        x_test, y_test: 3D inputs for LSTM testing
        """
        if target not in self.filedata_df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")
        if len(self.filedata_df) < timesteps:
            raise ValueError(f"Not enough data points to create sequences with timesteps={timesteps}.")

        self.filedata_df.sort_index(inplace=True)
        self.filedata_df.ffill(inplace=True)
        self.filedata_df.bfill(inplace=True)

        feature_cols = [col for col in self.filedata_df.columns if col != target]

        # Split into training and test sets
        train_size = int((1 - test_size) * len(self.filedata_df))
        df_train = self.filedata_df[:train_size]
        df_test = self.filedata_df[train_size:]

        # Prepare LSTM-specific 3D data [samples, timesteps, features]
        x_train = np.array([df_train[feature_cols].values[i:i + timesteps] for i in range(len(df_train) - timesteps)])
        y_train = df_train[target].values[timesteps:]

        x_test = np.array([df_test[feature_cols].values[i:i + timesteps] for i in range(len(df_test) - timesteps)])
        y_test = df_test[target].values[timesteps:]

        print(f"x_train shape: {x_train.shape}")  # Should be (samples, timesteps, features)
        print(f"y_train shape: {y_train.shape}")  # Should be (samples,)
        print(f"x_test shape: {x_test.shape}")  # Should be (samples, timesteps, features)
        print(f"y_test shape: {y_test.shape}")  # Should be (samples,)

        return x_train, y_train, x_test, y_test

