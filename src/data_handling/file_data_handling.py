import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_handling.async_database import AsyncDatabase

class FileDataHandler:
    def __init__(self, api_connection, file_path):
        self.api_connection = api_connection
        self.file_path = file_path
        self.file_data = None
        self.filedata_df = None

    async def run(self):
        await self.fetch_data()
        self.process_data()

    async def fetch_data(self):
        """
                Fetch the file data including precomputed features from the database.
        """
        query = {"path": self.file_path}
        projection = {"commit_history": 1, "features": 1, "_id": 0}  # Fetch only relevant fields
        self.file_data = await AsyncDatabase.find(self.api_connection.file_tracking_collection, query, projection)
        if not self.file_data:
            raise ValueError(f"No data found for file: {self.file_path}")
        print(f"Fetched data for {self.file_path}: {self.file_data}")


    def process_data(self):
        commit_history = self.file_data[0].get("commit_history", [])
        features = self.file_data[0].get("features", [])

        # Create a DataFrame for commit history
        history_df = pd.DataFrame(commit_history)
        history_df["date"] = pd.to_datetime(history_df["date"])
        history_df.set_index("date", inplace=True)

        # Create a DataFrame for features
        features_df = pd.DataFrame(features)
        features_df["date"] = pd.to_datetime(features_df["date"])
        features_df.set_index("date", inplace=True)

        # Merge the two DataFrames on the date index
        self.filedata_df = history_df.merge(features_df, how="outer", left_index=True, right_index=True)

        # Ensure data is sorted and resampled daily
        self.filedata_df.sort_index(inplace=True)
        self.filedata_df = self.filedata_df.resample('D').ffill()  # Fill missing dates

        # Handle any remaining missing values if necessary
        self.filedata_df.ffill(inplace=True)
        self.filedata_df.bfill(inplace=True)

    def prepare_arima_data(self, target, test_size=0.2):
        if target not in self.filedata_df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        arima_df = self.filedata_df.copy()

        arima_df.sort_index(inplace=True)
        arima_df.dropna(subset=[target], inplace=True)

        feature_cols = [col for col in arima_df.columns if col != target]
        features = arima_df[feature_cols].values
        target_values = arima_df[target].values

        train_size = int((1 - test_size) * len(features))
        x_train, y_train = features[:train_size], target_values[:train_size]
        x_test, y_test = features[train_size:], target_values[train_size:]

        return x_train, y_train, x_test, y_test

    # Tried for Prophet last
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

        self.filedata_df.replace([np.inf, -np.inf], np.nan)
        self.filedata_df.dropna(subset=[target], inplace=True)
        self.filedata_df.sort_index(inplace=True)

        train, test = train_test_split(self.filedata_df, test_size=test_size, shuffle=False)

        x_train = train.index.tz_localize(None)
        y_train = train[target]

        x_test = test.index.tz_localize(None)
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

    def aggregate_cluster_features(self, combined_df, feature_name):
        cluster_time_series = {}

        for cluster_id in combined_df['cluster'].unique():
            cluster_files = combined_df[combined_df['cluster'] == cluster_id]

            file_features = []
            for _, row in cluster_files.iterrows():
                file1_features = self.file_features[self.file_features['path'] == row['file1']]
                file2_features = self.file_features[self.file_features['path'] == row['file2']]
                file_features.append(pd.concat([file1_features, file2_features]))

            # Combine and sort features by date
            cluster_feature_df = pd.concat(file_features).sort_values('date')

            # Aggregate the feature across all files in the cluster
            time_series = cluster_feature_df.groupby('date')[feature_name].mean()
            cluster_time_series[cluster_id] = time_series

        return cluster_time_series
