from src.data_handling.repodata_handler import RepoDataHandler
from src.predictions.model_training import ModelTrainer
from src.visualisations.plotting import Plotter


class CommitVisualiser:
    def __init__(self, api_connection, project_name, models, modeling_tasks):
        self.models = models
        self.modeling_tasks = modeling_tasks

        self.plotter = Plotter(project_name=project_name)
        self.data_handler = RepoDataHandler(api_connection, self.models, self.modeling_tasks)
        self.model_trainer = ModelTrainer(self.models, modeling_tasks=self.modeling_tasks)

        self.commits = None

    async def run(self):
        await self.data_handler.run()
        data_splits = self.data_handler.prepare_data()

        self.commits = self.data_handler.commit_data

        if self.models:
            for task, (x_train, y_train, x_test, y_test) in data_splits.items():
                print(f"Training and evaluating model for {task}")
                model_info = self.model_trainer.train_and_evaluate_model(x_train, y_train, x_test, y_test)
                self.plotter.plot_commit_predictions(self.data_handler.commits_df, model_info, task)
