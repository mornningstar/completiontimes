from src.data_handling.repodata_handler import RepoDataHandler
from src.predictions.machine_learning.lstmmodel import LSTMModel
from src.predictions.model_training import ModelTrainer
from src.predictions.statistical_predictions.arima import ARIMAModel
from src.predictions.statistical_predictions.sarima import SARIMAModel
from src.visualisations.plotting import Plotter


class CommitVisualiser:
    def __init__(self, api_connection, project_name, model_classes, modeling_tasks):
        self.model_classes = model_classes
        self.modeling_tasks = modeling_tasks

        self.plotter = Plotter(project_name=project_name)
        self.data_handler = RepoDataHandler(api_connection, self.model_classes, self.modeling_tasks)
        self.model_trainer = ModelTrainer(self.model_classes, modeling_tasks=self.modeling_tasks)

        self.commits = None

    async def run(self):
        await self.data_handler.run()

        arima_data_splits = None
        lstm_data_splits = None
        other_data_splits = None

        if any(model_class in [ARIMAModel, SARIMAModel] for model_class in self.model_classes):
            arima_data_splits = self.data_handler.prepare_arima_data()
        if any(model_class == LSTMModel for model_class in self.model_classes):
            lstm_data_splits = self.data_handler.prepare_lstm_data()
        if any(model_class not in [ARIMAModel, SARIMAModel, LSTMModel] for model_class in self.model_classes):
            other_data_splits = await self.data_handler.prepare_data()

        self.commits = self.data_handler.commit_data

        for model in self.model_classes:
            if model in [ARIMAModel, SARIMAModel] and arima_data_splits:
                data_splits = arima_data_splits
            elif model == LSTMModel and lstm_data_splits:
                data_splits = lstm_data_splits
            else:
                data_splits = other_data_splits

            for task, (x_train, y_train, x_test, y_test) in data_splits.items():

                print(f"Training and evaluating model for {task}")
                model_info = self.model_trainer.train_and_evaluate_model(x_train, y_train, x_test, y_test)
                print("ended training and evaluating model")

                if model == LSTMModel:
                    print("Started plotting LSTM predictions")
                    self.plotter.plot_lstm_predictions(self.data_handler.commits_df, model_info, task)
                    print("Ended plotting LSTM predictions")
                elif model in [ARIMAModel, SARIMAModel]:
                    print("Started plotting ARIMA/SARIMA predictions")
                    self.plotter.plot_commit_predictions(self.data_handler.commits_df, model_info, task)
                    print("Ended plotting ARIMA/SARIMA predictions")
                else:
                    print(f"Started plotting predictions for {model}")
                    self.plotter.plot_commit_predictions(self.data_handler.commits_df, model_info, task)
                    print(f"Ended plotting predictions for {model}")