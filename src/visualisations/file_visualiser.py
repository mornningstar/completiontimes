import logging

import matplotlib
import pandas as pd

from src.data_handling.preprocessing.file_data_handling import FileDataHandler
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

        self.model_trainer = ModelTrainer(models)
        self.plotter = Plotter(self.project_name)
        self.file_path = file_path
        self.targets = targets
        self.commit_visualiser = commit_visualiser
        self.all_file_features = all_file_features

        self.cluster_combined_df = cluster_combined_df
        self.cluster = self.cluster_combined_df is not None

        self.data_handler = FileDataHandler(self.api_connection, file_path, targets, all_file_features,
                                            cluster_combined_df, cluster=self.cluster)

        self.model_info = None

    def prepare_data(self, target, series=None, cluster: bool =False):

        if series is None:
            if not hasattr(self.data_handler, "filedata_df") or self.data_handler.filedata_df.empty:
                raise ValueError("No valid data available to prepare.")
            series = self.data_handler.filedata_df

        if cluster:
            series = self.data_handler.prepare_cluster_data(target, series)

        self.logging.info(f"Series is:\n{series}")

        return self.data_handler.prepare_model_specific_data(self.models, target, series)

    async def run(self):
        self.logging.info("Starting file visualiser")

        if not self.cluster:
            await self.data_handler.process_data()
            for target in self.targets:

                x_train, x_test, y_train, y_test = self.prepare_data(target)
                self.model_info = self.model_trainer.train_and_evaluate_model(x_train, y_train, x_test, y_test)

                self.plotter.plot_predictions(self.data_handler.filedata_df, self.model_info, self.file_path, target)

        elif self.cluster and self.cluster_combined_df is not None:

            if self.cluster_combined_df.empty or "cluster" not in self.cluster_combined_df.columns:
                raise ValueError(
                    "Cluster assignments (self.cluster_combined_df with 'cluster' column) are required for cluster mode."
                )

            print("Calling agg cluster")
            print(type(self.cluster_combined_df))
            pd.set_option('display.max_columns', None)
            print(self.cluster_combined_df[:4])

            cluster_time_series = await self.data_handler.process_data()
            print("CTS: ", cluster_time_series)
            aggregated_clusters = self.data_handler.aggregate_cluster_features(cluster_time_series)
            print("AC: ", aggregated_clusters)

            for cluster_id, series in aggregated_clusters.groupby("cluster"):
                for target in self.targets:
                    self.logging.info(f"Processing Cluster {cluster_id} for target: {target}")

                    x_train, x_test, y_train, y_test = self.prepare_data(target, series, cluster=True)
                    self.model_info = self.model_trainer.train_and_evaluate_model(x_train, y_train, x_test, y_test)

                    self.plotter.plot_predictions(series, self.model_info, f"Cluster {cluster_id} {target}", target)

    async def predict_completion(self, target, steps, threshold=10, consecutive_days=7):
        """
        Predict file mixins using a trained model.

        :param target: The target feature for mixins prediction (e.g., 'cumulative_size').
        :param steps: Number of future steps to predict.
        :param threshold: Percentage change threshold for mixins.
        :param consecutive_days: Number of consecutive days the threshold must be met.
        :return: Predicted mixins date or None if not met within the horizon.
        """

        if self.cluster:
            # In cluster mode, iterate over each cluster in the aggregated DataFrame.
            predictions_by_cluster = {}
            # Group by the 'cluster' column
            for cluster_id, cluster_df in self.data_handler.clusterdata_df.groupby("cluster"):
                if target not in cluster_df.columns:
                    self.logging.warning(f"Target {target} not found for cluster {cluster_id}. Skipping.")
                    continue

                full_data = cluster_df[target]
                # You may want to check that full_data has enough points
                if len(full_data) < 2:
                    self.logging.warning(f"Not enough data for cluster {cluster_id} to predict mixins.")
                    continue

                # Use the model (assumed to be trained already) for prediction.
                # Here, we use the same model for all clusters; if you need separate models, update accordingly.
                model = next(iter(self.model_info.values()))["model"]

                full_x_train = full_data.index
                full_y_train = full_data.values

                future_dates, pred = self.model_trainer.refit_model(model, full_x_train, full_y_train, steps=steps)

                model_info = self.model_info[model.__class__.__name__]
                model_info["refit_future_dates"] = future_dates
                model_info["refit_predictions"] = pred

                self.plotter.plot_refit_predictions(full_data, future_dates, pred, cluster_id, target)

                percentage_changes = [
                    (pred[i] - pred[i - 1]) / pred[i - 1] * 100 for i in range(1, len(pred))
                ]

                completion_date = None
                for i in range(len(percentage_changes) - consecutive_days + 1):
                    window = percentage_changes[i:i + consecutive_days]
                    if all(abs(change) < threshold for change in window):
                        completion_days = i + consecutive_days
                        completion_date = full_data.index[-1] + pd.Timedelta(days=completion_days)
                        self.logging.info(
                            f"Predicted mixins date for cluster {cluster_id}, target {target}: {completion_date}"
                        )
                        break

                if completion_date is not None:
                    predictions_by_cluster[cluster_id] = completion_date
                else:
                    self.logging.info(f"No mixins detected for cluster {cluster_id} within {steps} days.")

            return predictions_by_cluster

        else:
            # Non-cluster (single file) mode
            data_df = self.data_handler.filedata_df
            if target not in data_df.columns:
                raise ValueError(f"Target {target} not found in filedata_df.")

            full_data = data_df[target]
            model = next(iter(self.model_info.values()))["model"]
            full_x_train = full_data.index
            full_y_train = full_data.values

            future_dates, predictions_ = self.model_trainer.refit_model(model, full_x_train, full_y_train, steps=steps)

            percentage_changes = [
                (predictions_[i] - predictions_[i - 1]) / predictions_[i - 1] * 100
                for i in range(1, len(predictions_))
            ]

            for i in range(len(percentage_changes) - consecutive_days + 1):
                window = percentage_changes[i:i + consecutive_days]
                if all(abs(change) < threshold for change in window):
                    completion_days = i + consecutive_days
                    completion_date = full_data.index[-1] + pd.Timedelta(days=completion_days)
                    self.logging.info(f"Predicted mixins date for {target}: {completion_date}")
                    return completion_date

            self.logging.info(f"No mixins detected for {target} within {steps} days.")
            return None
