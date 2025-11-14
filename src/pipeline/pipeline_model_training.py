import os
from typing import List

from src.factories.trainer_factory import TrainerFactory
from src.pipeline.configs import TRAINER_BY_TYPE
from src.visualisations.model_plotting import ModelPlotter


class ModelTrainingPipeline:
    def __init__(self, project_name, models: List[dict], feature_pipeline, images_dir: str, models_dir: str):
        self.project_name = project_name
        self.models = models
        self.feature_pipeline = feature_pipeline
        self.images_dir = images_dir
        self.models_dir = models_dir

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

            trainer.train_and_evaluate(features_to_use)
            trainer.predict_unlabeled_files(features_to_use[0], latest_only=True)
