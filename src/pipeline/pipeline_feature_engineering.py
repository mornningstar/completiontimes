import logging
from typing import Dict, Tuple, Type

import pandas as pd

from src.data_handling.features.feature_engineer_runner import FeatureEngineerRunner
from src.data_handling.features.regression_feature_eng import RegressionFeatureEngineering
from src.data_handling.features.survival_feature_engineer import SurvivalFeatureEngineer

ENGINEER_BY_TYPE = {
    "regression": RegressionFeatureEngineering,
    "survival":   SurvivalFeatureEngineer,
}

class FeatureEngineeringPipeline:
    def __init__(self, file_repo, plotter, source_directory):
        self.file_repo = file_repo
        self.plotter = plotter
        self.source_directory = source_directory
        self._features_cache: Dict[Tuple[Type, bool], pd.DataFrame] = {}

    async def get_or_create_features(self, model_cfg: dict):
        flag = model_cfg.get("use_categorical", False)
        feature_type = model_cfg.get("feature_type", "regression")
        eng_cls = ENGINEER_BY_TYPE[feature_type]

        cache_key = (eng_cls, flag)
        if cache_key not in self._features_cache:
            engineer = eng_cls(self.file_repo, self.plotter, self.source_directory)
            runner = FeatureEngineerRunner(engineer)
            engineered_df = await runner.run(source_directory=self.source_directory)
            self._features_cache[cache_key] = engineered_df
            logging.info(
                f"Computed features with {eng_cls.__name__} (use_categorical={flag}) - rows={len(engineered_df)}"
            )

        return self._features_cache[cache_key]

    async def run(self, model_cfg: dict):
        return await self.get_or_create_features(model_cfg)