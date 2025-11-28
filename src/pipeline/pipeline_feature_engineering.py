import logging
from typing import Dict, Tuple, Type

import pandas as pd

from src.data_handling.features.feature_engineer_runner import FeatureEngineerRunner
from src.pipeline.configs import ENGINEER_BY_TYPE


class FeatureEngineeringPipeline:
    def __init__(self, file_repo, plotter, source_directory, labelling_config=None):
        self.file_repo = file_repo
        self.plotter = plotter
        self.source_directory = source_directory
        self.labelling_config = labelling_config
        self._features_cache: Dict[Tuple[Type, bool], pd.DataFrame] = {}

    async def get_or_create_features(self, model_cfg: dict):
        feature_type = model_cfg.get("feature_type", "regression")
        eng_cls = ENGINEER_BY_TYPE[feature_type]

        cache_key = eng_cls
        if cache_key not in self._features_cache:
            engineer = eng_cls(self.file_repo, self.plotter, labelling_config=self.labelling_config)
            runner = FeatureEngineerRunner(engineer)
            engineered_df = await runner.run(source_directory=self.source_directory)
            self._features_cache[cache_key] = engineered_df
            logging.info(
                f"Computed features with {eng_cls.__name__} - rows={len(engineered_df)}"
            )

        return self._features_cache[cache_key]

    async def run(self, model_cfg: dict):
        return await self.get_or_create_features(model_cfg)
