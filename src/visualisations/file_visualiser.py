import logging

import matplotlib
import pandas as pd

from src.data_handling.file_data_handling import FileDataHandler
from src.predictions.model_training import ModelTrainer
from src.visualisations.plotting import Plotter

matplotlib.use('Agg')


class FileVisualiser:
    def __init__(self, api_connection, project_name,file_path, commit_visualiser, models, targets, all_file_features,
                 cluster_combined_df=None):
        self.logging = logging.getLogger(self.__class__.__name__)

        self.models = models
        self.api_connection = api_connection
        self.project_name = project_name
        self.data_handler = FileDataHandler(self.api_connection, file_path, targets, all_file_features)
        self.model_trainer = ModelTrainer(models)
        self.plotter = Plotter(self.project_name)
        self.file_path = file_path
        self.targets = targets
        self.commit_visualiser = commit_visualiser
        self.all_file_features = all_file_features

        self.cluster_combined_df = cluster_combined_df
        self.cluster = self.cluster_combined_df is not None

        self.model_info = None

    def prepare_data(self, target, series=None, cluster=False):

        if series is None:
            if not hasattr(self.data_handler, "filedata_df") or self.data_handler.filedata_df.empty:
                raise ValueError("No valid data available to prepare.")
            series = self.data_handler.filedata_df

        if cluster:
            series = self.data_handler.prepare_cluster_data(series, target)

        self.logging.info(f"Series is:\n{series}")

        return self.data_handler.prepare_model_specific_data(self.models, target, series)

    async def run(self):
        self.logging.info("Starting file visualiser")
        if not self.cluster:
            await self.data_handler.run(cluster=False)
            for target in self.targets:

                x_train, y_train, x_test, y_test = self.prepare_data(target)
                self.model_info = self.model_trainer.train_and_evaluate_model(x_train, y_train, x_test, y_test)

                self.plotter.plot_predictions(self.data_handler.filedata_df, self.model_info, self.file_path, target)

        elif self.cluster and self.cluster_combined_df is not None:

            if not self.cluster_combined_df or "cluster" not in self.cluster_combined_df.columns:
                raise ValueError(
                    "Cluster assignments (self.cluster_combined_df with 'cluster' column) are required for cluster mode."
                )

            cluster_time_series = self.data_handler.aggregate_cluster_features(self.cluster_combined_df)
            await self.data_handler.run(cluster=True, cluster_time_series=cluster_time_series)

            for cluster_id, series in cluster_time_series.items():
                for target in self.targets:
                    self.logging.info(f"Processing Cluster {cluster_id} for target: {target}")

                    x_train, x_test, y_train, y_test = self.prepare_data(target, series, cluster=True)
                    model_info = self.model_trainer.train_and_evaluate_model(x_train, y_train, x_test, y_test)
                    self.plotter.plot_predictions(series, model_info, f"Cluster {cluster_id} {target}", target)

    async def predict_completion(self, target, steps, threshold=10, consecutive_days=7):
        """
        Predict file completion using a trained model.

        :param target: The target feature for completion prediction (e.g., 'cumulative_size').
        :param steps: Number of future steps to predict.
        :param threshold: Percentage change threshold for completion.
        :param consecutive_days: Number of consecutive days the threshold must be met.
        :return: Predicted completion date or None if not met within the horizon.
        """
        # Ensure the target exists
        if target not in self.data_handler.filedata_df.columns:
            raise ValueError(f"Target {target} not found in filedata_df.")

        # Get full data
        full_data = self.data_handler.filedata_df[target]

        #model = self.model_info["model"]
        model = next(iter(self.model_info.values()))["model"]

        full_x_train = full_data.index
        full_y_train = full_data.values

        future_dates, predictions = self.model_trainer.refit_model(model, full_x_train, full_y_train, steps=steps)

        percentage_changes = [
            (predictions[i] - predictions[i - 1]) / predictions[i - 1] * 100
            for i in range(1, len(predictions))
        ]

        for i in range(len(percentage_changes) - consecutive_days + 1):
            window = percentage_changes[i:i + consecutive_days]
            if all(abs(change) < threshold for change in window):
                completion_days = i + consecutive_days
                completion_date = full_data.index[-1] + pd.Timedelta(days=completion_days)
                self.logging.info(f"Predicted completion date for {target}: {completion_date}")
                return completion_date

        self.logging.info(f"No completion detected for {target} within {steps} days.")
        return None
