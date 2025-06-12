import logging

import numpy as np
import pandas as pd

from src.data_handling.clustering.cluster_aggregator import ClusterAggregator
from src.predictions.deep_learning.lstmmodel import LSTMModel
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
    :param target_types: str - The type of the target feature (e.g., 'cumulative', 'rolling').
    :return: pd.DataFrame or pd.Series - The gap-handled data.
    """
    if not isinstance(target_types, list):
        target_types = [target_types]  # Ensure target_types is always a list

    for target_type in target_types:
        method = GAP_HANDLING_METHODS.get(target_type, None)

        if not method:
            logging.warning(f"Skipping gap handling for {target_type}: No method defined.")
            return data

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
            data = data.resample('D').asfreq(fill_value=0)
        elif method is None:
            raise ValueError(f"No gap handling method defined for target type: {target_type}")
        else:
            raise ValueError(f"Unknown gap handling method: {method}")

    return data


def infer_raw_target(cluster_target: str) -> str:
    """
    Infers the base metric from a cluster-level target.

    For example:
      - If the target is 'cluster_cumulative_size', return 'size'
      - Otherwise, if the target starts with 'cluster_', remove that prefix.
    """
    if "cumulative" in cluster_target:
        # For cumulative metrics, we typically want to aggregate the raw 'size'
        return "size"
    elif cluster_target.startswith("cluster_"):
        return cluster_target[len("cluster_"):]
    else:
        return cluster_target


class FileDataHandler:
    def __init__(self, api_connection, file_path, targets, all_file_features, cluster_combined_df, cluster):
        self.logging = logging.getLogger(self.__class__.__name__)
        self.api_connection = api_connection
        self.file_path = file_path
        self.targets = targets

        self.file_data = None
        self.filedata_df = None
        self.clusterdata_df = None
        self.all_file_features = all_file_features

        self.cluster_combined_df = cluster_combined_df

        self.cluster = cluster
        self.target_contains_cumulative = "cumulative" in self.targets[0]

        self.raw_target = infer_raw_target(self.targets[0]) if self.cluster else self.targets[0]


    async def process_data(self):
        print("Processing data in DataHandler")

        if not self.cluster:
            features_df = self.all_file_features[self.all_file_features["path"] == self.file_path].copy()
            features_df.index = pd.to_datetime(features_df.index).tz_localize(None)
            features_df.index = features_df.index.normalize()

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

        elif self.cluster:
            cluster_time_series = {}

            for cluster_id, cluster_files in self.cluster_combined_df.groupby("cluster"):

                file_paths = set(cluster_files["file1"]).union(set(cluster_files["file2"]))

                file_features = self.all_file_features[self.all_file_features["path"].isin(file_paths)].copy()

                if file_features.empty:
                    logging.warning(f"No valid files found for cluster {cluster_id}")
                    continue

                file_features.index = pd.to_datetime(file_features.index).normalize()
                min_date = file_features.index.min()
                max_date = file_features.index.max()
                full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")
                print(f"Cluster {cluster_id}: Date range from {min_date} to {max_date}")

                processed_files = []

                for file_path in file_paths:
                    file_data = file_features[file_features["path"] == file_path].copy()

                    if file_data.empty:
                        print(f"No data found for file {file_path} in cluster {cluster_id}")

                    if file_data.index.duplicated().any():
                        logging.warning(f"Duplicate timestamps found for {file_path}. Aggregating values.")
                        numeric_columns = file_data.select_dtypes(include=["number"]).columns
                        file_data = file_data.groupby(file_data.index)[numeric_columns].max()

                    file_data = file_data.reindex(full_date_range)
                    print(f"File {file_path}: Reindexed, now has {len(file_data)} rows.")

                    file_data[self.raw_target] = handle_gaps(file_data[self.raw_target], self.raw_target)

                    for target in [self.raw_target]:
                        first_valid_index = file_data[target].first_valid_index()
                        if first_valid_index is not None:
                            file_data.loc[file_data.index < first_valid_index, target] = 0

                    file_data["path"] = file_path

                    processed_files.append(file_data)

                # Combine all processed files for this cluster
                cluster_time_series[cluster_id] = pd.concat(processed_files)

                if not processed_files:
                    print(f"No processed files for cluster {cluster_id}!")
                else:
                    print(f"Cluster {cluster_id}: Processed {len(processed_files)} files.")

            return cluster_time_series

    def validate_target(self, target):
        if not self.cluster:
            if self.filedata_df is None or self.filedata_df.empty:
                raise ValueError("File data is not available!")
            if target not in self.filedata_df.columns:
                raise ValueError(f"Target column '{target}' not found in filedata_df.")
        else:
            if self.clusterdata_df is None or self.clusterdata_df.empty:
                raise ValueError("Cluster data is not available!")
            if target not in self.clusterdata_df.columns:
                raise ValueError(f"Target column '{target}' not found in clusterdata_df.")

    def prepare_model_specific_data(self, models, target, data, timesteps=10, test_size=0.2):
        self.logging.info(f"Preparing data for target: {target}")
        if any(model == LSTMModel for model in models):
            return self.prepare_lstm_data(target, data, timesteps, test_size)
        elif any(model == SeasonalARIMABase for model in models):
            return self.prepare_arima_data(target, data, test_size)
        else:
            return self.prepare_data(target, data, test_size=test_size)

    def prepare_arima_data(self, target, data, test_size=0.2):
        self.logging.debug("Prepare ARIMA data in DataHandler")
        self.validate_target(target)

        arima_df = data.copy()

        arima_df[target] = pd.to_numeric(arima_df[target], errors='coerce')
        arima_df.dropna(subset=[target], inplace=True)

        arima_df.sort_index(inplace=True)

        dates = arima_df.index
        target_values = arima_df[target].values

        self.logging.debug(f"Dates values: {dates[:5]}")
        self.logging.debug(f"Target values: {target_values[:5]} (dtype: {target_values.dtype})")

        train_size = int((1 - test_size) * len(target_values))
        if train_size == 0:
            raise ValueError("Not enough data to split into train and test sets.")

        x_train = dates[:train_size]
        y_train = target_values[:train_size]
        x_test = dates[train_size:]
        y_test = target_values[train_size:]

        print("x_train: ", x_train)
        print("y_train: ", y_train)
        print("x_test: ", x_test)
        print("y_test: ", y_test)

        return x_train, y_train, x_test, y_test

    # Tried for Prophet last
    def prepare_data(self, target, data, test_size=0.2):
        """
        Prepares the data for visualisation.
        :param data:
        :param target: target column - which column to predict
        :param test_size: Percentage of data to be used for testing
        :return:
            x_train: timestamps of train datatest,
            x_test: timestamps of test dataset,
            y_train: file sizes of train dataset,
            y_test: file sizes of test dataset
        """
        self.validate_target(target)

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=[target], inplace=True)
        data.sort_index(inplace=True)

        train_size = int(len(data) * (1 - test_size))

        train = data.iloc[:train_size].copy()  # Ensure copy
        test = data.iloc[train_size:].copy()

        x_train = train.index.tz_localize(None)
        y_train = train[target].values

        x_test = test.index.tz_localize(None)
        y_test = test[target].values

        print(f"x_train length: {len(x_train)}, y_train length: {len(y_train)}")
        print(f"x_test length: {len(x_test)}, y_test length: {len(y_test)}")

        return x_train, x_test, y_train, y_test

    def prepare_lstm_data(self, target, data, timesteps=10, test_size=0.2):
        """
        Prepares data for LSTM models (3D input for LSTM).
        :param data:
        :param target: target column - which column to predict
        :param timesteps: Number of time steps to consider in the LSTM input
        :param test_size: Percentage of data to be used for testing
        :return:
        x_train, y_train: 3D inputs for LSTM training,
        x_test, y_test: 3D inputs for LSTM testing
        """
        self.validate_target(target)

        if len(data) < timesteps:
            raise ValueError(f"Not enough data points to create sequences with timesteps={timesteps}.")

        data.sort_index(inplace=True)
        data.ffill(inplace=True)
        data.bfill(inplace=True)

        feature_cols = [col for col in data.columns if col != target]

        # Split into training and test sets
        train_size = int((1 - test_size) * len(data))
        df_train = data[:train_size]
        df_test = data[train_size:]

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

        cluster_data = cluster_data.copy()
        cluster_data.dropna(subset=[target], inplace=True)
        cluster_data.sort_index(inplace=True)

        return cluster_data

    def aggregate_cluster_features(self, cluster_time_series):
        aggregator = ClusterAggregator(self.raw_target, self.target_contains_cumulative)
        aggregated_clusters_df = aggregator.aggregate_cluster_features(cluster_time_series)
        self.clusterdata_df = aggregated_clusters_df

        return aggregated_clusters_df
