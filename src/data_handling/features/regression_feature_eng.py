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

    def engineer_features(self, file_df, window=7, include_sets = None):
        file_df = super().engineer_features(file_df, window, include_sets)

        first_commit = file_df.groupby("path")["date"].transform("min")
        file_df["age_in_days"] = (file_df["date"] - first_commit).dt.days
        file_df["commits_per_day_so_far"] = file_df["total_commits"] / (file_df["age_in_days"] + 1)
        file_df["growth_x_age"] = file_df["recent_growth_ratio"] * file_df["age_in_days"]

        file_df = self.add_days_until_completion(file_df)

        numeric_cols = [col for col in file_df.select_dtypes(include=[np.number]).columns
                        if col != "days_until_completion"]

        file_df[numeric_cols] = file_df[numeric_cols].fillna(0.0)

        return file_df
