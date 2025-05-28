import logging

import numpy as np

from src.data_handling.features.base_feature_engineer import BaseFeatureEngineer


class RegressionFeatureEngineering(BaseFeatureEngineer):
    def __init__(self, file_repo, plotter, use_categorical):
        super().__init__(file_repo, plotter, use_categorical)
        self.logging = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def get_target_columns():
        return ["days_until_completion"]

    def calculate_metrics(self, file_df, window=7):
        file_df = super().calculate_metrics(file_df, window)
        file_df = self.add_days_until_completion(file_df)

        numeric_cols = [col for col in file_df.select_dtypes(include=[np.number]).columns
                        if col != "days_until_completion"]

        file_df[numeric_cols] = file_df[numeric_cols].fillna(0.0)

        return file_df