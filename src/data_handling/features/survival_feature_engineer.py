import logging

import numpy as np
import pandas as pd

from src.data_handling.features.base_feature_engineer import BaseFeatureEngineer


class SurvivalFeatureEngineer(BaseFeatureEngineer):
    def __init__(self, file_repo, plotter, use_categorical):
        super().__init__(file_repo, plotter, use_categorical)
        self.logging = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def get_target_columns():
        return ["duration", "event"]

    def calculate_metrics(self, df, window = 7):
        df = super().calculate_metrics(df)
        df = self._add_survival_targets(df)

        numeric_cols = [c for c in df.select_dtypes("number").columns if c not in ["duration", "event"]]
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

        return df

    def _add_survival_targets(self, df):
        today = pd.Timestamp.utcnow().normalize()
        df = df.sort_values(["path", "date"])

        df["end_date"] = df.groupby("path")["date"].shift(-1)
        df["end_date"] = df["end_date"].where(df["end_date"].notna(), df["completion_date"])
        df["end_date"] = df["end_date"].fillna(today)

        df["duration"] = (df["end_date"] - df["date"]).dt.days

        df["event"] = 0

        df["is_last"] = df.groupby("path")["date"].transform("max") == df["date"]
        df["event"] = (
                df["is_last"] & df["completion_reason"].isin(["stable_line_change", "idle_timeout"])
        ).astype(int)
        df.drop(columns="is_last", inplace=True)

        mask_completed = df["date"] == df["completion_date"]
        df.loc[mask_completed, "event"] = df.loc[mask_completed, "completion_reason"].isin(
            ["stable_line_change", "idle_timeout"]).astype(int)

        df = df[df["duration"] >= 0].copy()
        
        return df
