import numpy as np
import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator

@feature_generator_registry.register
class TemporalDynamicsFeatureGenerator(AbstractFeatureGenerator):
    def get_feature_names(self) -> list[str]:
        return [
            "commit_interval_days", "interval_entropy", "committer_entropy", "add_entropy", "deletions_entropy",
            "early_growth", "total_growth_so_far", "normalized_early_growth", "recent_sum", "recent_contribution_ratio",
            "commit_interval_std"
        ]

    def generate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df["commit_interval_days"] = df.groupby("path")["date"].diff().dt.days.fillna(0)

        def entropy(series):
            counts = series.value_counts(normalize=True)
            return -np.sum(counts * np.log2(counts + 1e-10))  # 1e-10 for numerical stability

        df["interval_entropy"] = df.groupby("path")["commit_interval_days"].transform(entropy)
        df["committer_entropy"] = df.groupby("path")["committer"].transform(entropy)

        df["add_entropy"] = df.groupby("path")["lines_added"].transform(entropy)
        df["deletions_entropy"] = df.groupby("path")["lines_deleted"].transform(entropy)

        df = self._add_normalized_early_growth(df, early_n=7)
        df = self._add_recent_contribution_ratio(df, n=5)

        df["commit_interval_std"] = (
            df.groupby("path")["commit_interval_days"].transform("std").fillna(0)
        )

        return df

    def _add_normalized_early_growth(self, df, early_n=7):
        df["early_growth"] = (
            df.groupby("path")["size_diff"]
            .transform(lambda x: x.rolling(window=early_n, min_periods=1).sum())
        )

        df["total_growth_so_far"] = df.groupby("path")["size_diff"].cumsum() + 1e-10  # Avoid div by 0

        df["normalized_early_growth"] = (
                df["early_growth"] / df["total_growth_so_far"]
        ).clip(0, 1)

        return df

    def _add_recent_contribution_ratio(self, df, n=5):
        df["recent_sum"] = (
            df.groupby("path")["size_diff"]
            .transform(lambda x: x.rolling(window=n, min_periods=1).sum())
        )

        df["recent_contribution_ratio"] = (df["recent_sum"] / df["total_growth_so_far"]).clip(0, 1)

        return df
