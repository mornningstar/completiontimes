import logging
import os
from typing import Dict, Any

from src.pipeline.configs import TRAINER_BY_TYPE
from src.predictions.training.model_evaluator import ModelEvaluator
from src.visualisations.model_plotting import ModelPlotter


class TrainerFactory:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_trainer(self, project_name: str, model_cfg: Dict[str, Any],
                       images_dir:str, models_dir: str, ablation_name: str = None):
        feature_type = model_cfg.get("feature_type", "regression")
        data_split = model_cfg.get("split_strategy", "by_file")
        model_name = model_cfg["class"].__name__

        model_specific_images_dir = os.path.join(images_dir, model_name, data_split)
        model_specific_model_dir = os.path.join(models_dir, model_name, data_split)

        if ablation_name:
            model_specific_images_dir = os.path.join(model_specific_images_dir, ablation_name)
            model_specific_model_dir = os.path.join(model_specific_model_dir, ablation_name)

        os.makedirs(model_specific_images_dir, exist_ok=True)
        os.makedirs(model_specific_model_dir, exist_ok=True)

        trainer_cls = TRAINER_BY_TYPE.get(feature_type)
        if not trainer_cls:
            raise ValueError(f"No trainer found for feature_type: {feature_type}")

        self.logger.info(f"Instantiating dependencies for {model_name} ({data_split})")

        model_cls = model_cfg["class"]
        model = model_cls(auto_tune=True)

        model_plotter = ModelPlotter(project_name, images_dir=model_specific_images_dir)

        evaluator_logger = logging.getLogger(ModelEvaluator.__name__)
        evaluator = ModelEvaluator(
            model=model,
            output_dir=model_specific_model_dir,
            logger=evaluator_logger
        )

        trainer = trainer_cls(
            project_name=project_name,
            model_cfg=model_cfg,
            model_plotter=model_plotter,
            evaluator=evaluator,
            output_dir=model_specific_model_dir
        )

        self.logger.info(f"Successfully created trainer: {model_name}")
        return trainer