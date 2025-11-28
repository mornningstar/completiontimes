import logging

import numpy as np

from src.data_handling.features.base_feature_engineer import BaseFeatureEngineer


class RegressionFeatureEngineering(BaseFeatureEngineer):
    def __init__(self, file_repo, plotter, labelling_config=None):
        super().__init__(file_repo, plotter, labelling_config=labelling_config)
        self.logging = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def get_target_columns():
        return ["days_until_completion"]

    def engineer_features(self, file_df, window=7, include_sets = None):
        file_df, categorical_cols = super().engineer_features(file_df, window, include_sets)
        file_df = self.completion_labler.add_days_until_completion(file_df)

        if file_df.empty:
            return file_df, categorical_cols

        numeric_cols = [col for col in file_df.select_dtypes(include=[np.number]).columns
                        if col != "days_until_completion"]

        file_df[numeric_cols] = file_df[numeric_cols].fillna(0.0)

        return file_df, categorical_cols
