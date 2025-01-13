import matplotlib

from src.data_handling.file_cooccurence_analyser import FileCooccurenceAnalyser
from src.data_handling.file_data_handling import FileDataHandler
from src.predictions.machine_learning.lstmmodel import LSTMModel
from src.predictions.model_training import ModelTrainer
from src.predictions.statistical_predictions.seasonal_arima_base import SeasonalARIMABase
from src.visualisations.plotting import Plotter

matplotlib.use('Agg')


class FileVisualiser:
    def __init__(self, api_connection, project_name,file_path, commit_visualiser, models, targets, all_file_features,
                 cluster_combined_df=None):
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

    def prepare_data(self, target, series=None):
        if series is None:
            series = self.data_handler.filedata_df

        if any(isinstance(model, LSTMModel) for model in self.models):
            return self.data_handler.prepare_lstm_data(series, target)
        elif any(isinstance(model, SeasonalARIMABase) for model in self.models):
            return self.data_handler.prepare_arima_data(series, target)
        else:
            return self.data_handler.prepare_data(series, target)


    async def run(self, mode="file"):
        await self.data_handler.run()

        if mode == "file":
            for target in self.targets:
                print(f"Processing target: {target}")

                x_train, x_test, y_train, y_test = self.prepare_data(target)
                model_info = self.model_trainer.train_and_evaluate_model(x_train, y_train, x_test, y_test)

                self.plotter.plot_predictions(self.data_handler.filedata_df, model_info, self.file_path, target)

        if mode == "cluster" and self.cluster_combined_df is not None:
            if "cluster" not in self.cluster_combined_df.columns:
                raise ValueError("Cluster assignments are required for cluster mode.")

            for target in self.targets:
                cluster_time_series = self.data_handler.aggregate_cluster_features(
                    self.cluster_combined_df, feature_name=target
                )

                for cluster_id, series in cluster_time_series.items():
                    print(f"Processing Cluster {cluster_id} for target: {target}")

                    x_train, x_test, y_train, y_test = self.prepare_data(target, series)
                    model_info = self.model_trainer.train_and_evaluate_model(x_train, y_train, x_test, y_test)
                    self.plotter.plot_predictions(series, model_info, f"Cluster {cluster_id} {target}", target)
