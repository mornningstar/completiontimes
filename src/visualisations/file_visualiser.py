import logging

import matplotlib

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
        self.data_handler = FileDataHandler(self.api_connection, file_path)
        self.model_trainer = ModelTrainer(models)
        self.plotter = Plotter(self.project_name)
        self.file_path = file_path
        self.targets = targets
        self.commit_visualiser = commit_visualiser
        self.all_file_features = all_file_features

        self.cluster_combined_df = cluster_combined_df
        self.cluster = self.cluster_combined_df is not None

    def prepare_data(self, target, series=None, cluster=False):
        if series is None:
            if not hasattr(self.data_handler, "filedata_df") or self.data_handler.filedata_df.empty:
                raise ValueError("No valid data available to prepare.")
            series = self.data_handler.filedata_df

        if cluster:
            series = self.data_handler.prepare_cluster_data(series, target)

        return self.data_handler.prepare_model_specific_data(self.models, target, series)

    async def run(self, mode="file"):
        await self.data_handler.run()

        if not self.cluster:
            for target in self.targets:
                self.logging.info(f"Processing target: {target}")

                x_train, x_test, y_train, y_test = self.prepare_data(target)
                model_info = self.model_trainer.train_and_evaluate_model(x_train, y_train, x_test, y_test)

                self.plotter.plot_predictions(self.data_handler.filedata_df, model_info, self.file_path, target)

        elif self.cluster and self.cluster_combined_df is not None:
            if not self.cluster_combined_df or "cluster" not in self.cluster_combined_df.columns:
                raise ValueError(
                    "Cluster assignments (self.cluster_combined_df with 'cluster' column) are required for cluster mode."
                )

            cluster_time_series = self.data_handler.aggregate_cluster_features(self.cluster_combined_df)

            for cluster_id, series in cluster_time_series.items():
                for target in self.targets:
                    self.logging.info(f"Processing Cluster {cluster_id} for target: {target}")

                    x_train, x_test, y_train, y_test = self.prepare_data(target, series, cluster=True)
                    model_info = self.model_trainer.train_and_evaluate_model(x_train, y_train, x_test, y_test)
                    self.plotter.plot_predictions(series, model_info, f"Cluster {cluster_id} {target}", target)
