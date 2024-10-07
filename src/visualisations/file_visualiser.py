import matplotlib

from src.data_handling.file_data_handling import FileDataHandler
from src.predictions.model_training import ModelTrainer
from src.visualisations.plotting import Plotter

matplotlib.use('TkAgg')


class FileVisualiser:
    def __init__(self, collection_name, file_path, models):
        self.data_handler = FileDataHandler(collection_name, file_path)
        self.model_trainer = ModelTrainer(models)
        self.plotter = Plotter(collection_name=collection_name)

        self.file_path = file_path

    async def run(self):
        await self.data_handler.fetch_data()
        x_train, y_train, x_test, y_test = self.data_handler.prepare_data()
        model_info = self.model_trainer.train_and_evaluate_model(x_train, y_train, x_test, y_test)
        self.plotter.plot(self.data_handler.size_df, model_info, self.file_path)
