import os
from typing import List

from src.pipeline.configs import TRAINER_BY_TYPE


class ModelTrainingPipeline:
    def __init__(self, project_name, models: List[dict], feature_pipeline, images_dir: str, models_dir: str):
        self.project_name = project_name
        self.models = models
        self.feature_pipeline = feature_pipeline
        self.images_dir = images_dir
        self.models_dir = models_dir

    async def run(self):
        for model_cfg in self.models:
            features_to_use = await self.feature_pipeline.get_or_create_features(model_cfg)

            model_name = model_cfg["class"].__name__
            data_split = model_cfg.get("split_strategy", "by_file")
            feature_type = model_cfg.get("feature_type", "regression")
            trainer_cls = TRAINER_BY_TYPE[feature_type]

            model_specific_images_dir = os.path.join(self.images_dir, model_name, data_split)
            model_specific_model_dir = os.path.join(self.models_dir, model_name, data_split)

            trainer = trainer_cls(
                self.project_name,
                model_cfg,
                images_dir=model_specific_images_dir,
                output_dir=model_specific_model_dir
            )
            trainer.train_and_evaluate(features_to_use)
            trainer.predict_unlabeled_files(features_to_use, latest_only=True)
