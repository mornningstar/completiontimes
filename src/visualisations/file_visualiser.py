import matplotlib

from src.data_handling.file_data_handling import FileDataHandler
from src.predictions.machine_learning.lstmmodel import LSTMModel
from src.predictions.model_training import ModelTrainer
from src.visualisations.plotting import Plotter

matplotlib.use('TkAgg')


class FileVisualiser:
    def __init__(self, project_name, file_path, models):
        self.models = models
        self.data_handler = FileDataHandler(project_name, file_path)
        self.model_trainer = ModelTrainer(models)
        self.plotter = Plotter(project_name=project_name)

        self.file_path = file_path

    async def run(self):
        await self.data_handler.fetch_data()

        if any(isinstance(model, LSTMModel) for model in self.models):
            # LSTM-specific data preparation
            x_train, y_train, x_test, y_test = self.data_handler.prepare_lstm_data()
        else:
            # General data preparation
            x_train, y_train, x_test, y_test = self.data_handler.prepare_data()

        model_info = self.model_trainer.train_and_evaluate_model(x_train, y_train, x_test, y_test)

        if any(isinstance(model, LSTMModel) for model in self.models):
            self.plotter.plot_lstm_predictions(self.data_handler.size_df, model_info, self.file_path)
        else:
            self.plotter.plot_predictions(self.data_handler.size_df, model_info, self.file_path)
