import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.predictions.machine_learning.lstmmodel import LSTMModel
from src.predictions.statistical_predictions.seasonal_arima_base import SeasonalARIMABase

GAP_HANDLING_METHODS = {
    'size': 'ffill',
    'cumulative_size': 'ffill',
    'cumulative_mean': 'ffill',
    'cumulative_std': 'ffill',

    'rolling_7_mean': 'interpolate',
    'rolling_7_std': 'interpolate',
    'rolling_7_max': 'interpolate',
    'rolling_7_min': 'interpolate',
    'rolling_7_median': 'interpolate',
    'rolling_7_var': 'interpolate',

    'absolute_change': 'zero_fill',
    'percentage_change': 'interpolate',

    'ema_7': ''
}


def handle_gaps(data, target_types):
    """
    Handles gaps in time series data based on the target type.

    :param data: pd.DataFrame or pd.Series - The time series data with a datetime index.
    :param target_type: str - The type of the target feature (e.g., 'cumulative', 'rolling').
    :return: pd.DataFrame or pd.Series - The gap-handled data.
    """
    if not isinstance(target_types, list):
        target_types = [target_types]  # Ensure target_types is always a list

    for target_type in target_types:
        method = GAP_HANDLING_METHODS.get(target_type, None)

        if method == 'ffill':
            logging.info(f'Ffilling gaps for {target_type}')
            data = data.resample('D').ffill()
        elif method == 'bfill':
            logging.info(f'Bfilling gaps for {target_type}')
            data = data.resample('D').bfill()
        elif method == 'interpolate':
            logging.info(f'Interpolating gaps for {target_type}')
            data = data.resample('D').interpolate(method='linear')
        elif method == 'zero_fill':
            logging.info(f'Zero-filling gaps for {target_type}')
            data = data.resample('D').fillna(0)
        elif method is None:
            raise ValueError(f"No gap handling method defined for target type: {target_type}")
        else:
            raise ValueError(f"Unknown gap handling method: {method}")

    return data



class FileDataHandler:
    def __init__(self, api_connection, file_path, targets, all_file_features):
        self.logging = logging.getLogger(self.__class__.__name__)
        self.api_connection = api_connection
        self.file_path = file_path
        self.targets = targets

        self.file_data = None
        self.filedata_df = None
        self.clusterdata_df = None
        self.all_file_features = all_file_features

    async def run(self, cluster=False, cluster_time_series=None):
        print("Run in DataHandler")
        self.process_data(cluster=cluster, cluster_time_series=cluster_time_series)

    def process_data(self, cluster=False, cluster_time_series=None):
        print("Processing data in DataHandler")

        if not cluster:
            features_df = self.all_file_features[self.all_file_features["path"] == self.file_path].copy()
            features_df.index = pd.to_datetime(features_df.index).tz_localize(None)

            if features_df.index.duplicated().any():
                self.logging.warning("Duplicate timestamps found in features_df. Aggregating values.")
                numeric_columns = features_df.select_dtypes(include=["number"]).columns
                features_df = features_df.groupby(features_df.index.date)[numeric_columns].mean()
                features_df.index = pd.to_datetime(features_df.index)

            features_df.sort_index(inplace=True)
            relevant_features = [col for col in self.targets if col in features_df.columns]
            for feature in relevant_features:
                features_df[feature] = handle_gaps(features_df[feature], feature)

            self.filedata_df = features_df[relevant_features]

        elif cluster:
            if not cluster_time_series:
                raise ValueError("Cluster mode requires 'cluster_time_series' input.")

            cluster_dfs = []
            for cluster_id, time_series in cluster_time_series.items():
                cluster_df = pd.DataFrame(time_series)

                # Set index as datetime
                if 'date' in cluster_df.columns:
                    cluster_df.set_index(pd.to_datetime(cluster_df['date']).dt.tz_localize(None), inplace=True)
                    cluster_df.drop(columns=['date'], inplace=True)

                # Handle duplicates in cluster_df
                if cluster_df.index.duplicated().any():
                    self.logging.warning(f"Duplicate timestamps found for Cluster {cluster_id}. Aggregating values.")

                    numeric_columns = cluster_df.select_dtypes(include=["number"]).columns
                    cluster_df = cluster_df.groupby(cluster_df.index.date)[numeric_columns].mean()
                    cluster_df.index = pd.to_datetime(cluster_df.index)

                # Sort index and handle gaps for each feature
                cluster_df.sort_index(inplace=True)
                for feature in cluster_df.columns:
                    cluster_df[feature] = handle_gaps(cluster_df[feature], feature)

                cluster_df["cluster_id"] = cluster_id
                cluster_dfs.append(cluster_df)

            self.clusterdata_df = pd.concat(cluster_dfs)

    def validate_target(self, target):
        if target not in self.filedata_df.columns:
            raise ValueError(f"Target column '{target}' not found in the data.")

    def prepare_model_specific_data(self, models, target, data, timesteps=10, test_size=0.2):
        print("Preparing model specific data in Visualiser")
        if any(isinstance(model, LSTMModel) for model in models):
            return self.prepare_lstm_data(target, timesteps, test_size)
        elif any(isinstance(model, SeasonalARIMABase) for model in models):
            return self.prepare_arima_data(target, test_size)
        else:
            return self.prepare_data(target, test_size)

    def prepare_arima_data(self, target, test_size=0.2):
        print("Prepare ARIMA data in DataHandler")
        self.validate_target(target)

        print("Target in ARIMA:")
        print(target[:5])

        arima_df = self.filedata_df.copy()

        print(self.filedata_df.head(5))

        arima_df.sort_index(inplace=True)
        arima_df.dropna(subset=[target], inplace=True)

        dates = arima_df.index
        target_values = arima_df[target].values

        print(f"Dates values: {dates[:5]}")
        print(f"Target values: {target_values[:5]}")

        train_size = int((1 - test_size) * len(target_values))
        if train_size == 0:
            raise ValueError("Not enough data to split into train and test sets.")

        x_train = dates[:train_size]
        y_train = target_values[:train_size]
        x_test = dates[train_size:]
        y_test = target_values[train_size:]

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
        self.validate_target(target)

        self.filedata_df.replace([np.inf, -np.inf], np.nan, inplace=True)
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
        self.validate_target(target)

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

    def prepare_cluster_data(self, target, cluster_data):
        """
        Prepares cluster-level data with minimal preprocessing.
        :param target: Target column to predict.
        :param cluster_data: Aggregated cluster-level time-series data.
        :return: Preprocessed cluster_data DataFrame.
        """
        self.validate_target(target)

        cluster_data = cluster_data.copy()
        cluster_data.dropna(subset=[target], inplace=True)
        cluster_data.sort_index(inplace=True)

        return cluster_data

    def aggregate_cluster_features(self, combined_df, date_col="date"):
        cluster_time_series = {}

        if date_col not in self.filedata_df.columns:
            raise ValueError(f"Date column '{date_col}' not found in filedata_df.")

        for cluster_id, cluster_files in combined_df.groupby("cluster"):
            file_features = self.filedata_df[
                self.filedata_df["path"].isin(cluster_files["file1"].tolist() + cluster_files["file2"].tolist())
            ]

            if file_features.empty:
                self.logging.warning(f"No valid files found for cluster {cluster_id}")
                continue

            # Aggregate features by date
            aggregated_features = file_features.groupby(date_col)[self.targets].mean()
            cluster_time_series[cluster_id] = aggregated_features

        return cluster_time_series
