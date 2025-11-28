import datetime
from typing import List

from src.factories.trainer_factory import TrainerFactory
from src.pipeline.results_writer import append_to_master_results


class ModelTrainingPipeline:
    def __init__(self, project_name, models: List[dict], feature_pipeline, images_dir: str, models_dir: str,
                 master_results_path: str, timestamp):
        self.project_name = project_name
        self.models = models
        self.feature_pipeline = feature_pipeline
        self.images_dir = images_dir
        self.models_dir = models_dir
        self.master_results_path = master_results_path
        self.timestamp = timestamp

        self.factory = TrainerFactory()

    async def run(self):
        for model_cfg in self.models:
            features_to_use = await self.feature_pipeline.get_or_create_features(model_cfg)

            trainer = self.factory.create_trainer(
                project_name=self.project_name,
                model_cfg=model_cfg,
                images_dir=self.images_dir,
                models_dir=self.models_dir
            )

            training_result = trainer.train_and_evaluate(features_to_use)

            result_data = {
                "project": self.project_name,
                "model": model_cfg['class'].__name__,
                "split_strategy": model_cfg.get("split_strategy", "by_file"),
                "configuration": "ALL",
                "timestamp": self.timestamp,
                "metrics": vars(training_result.metrics)
            }

            append_to_master_results(result_data, self.master_results_path)

            #trainer.predict_unlabeled_files(features_to_use[0], latest_only=True)
