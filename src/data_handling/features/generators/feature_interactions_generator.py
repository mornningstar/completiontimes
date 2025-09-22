import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator

@feature_generator_registry.register
class FeatureInteractionsGenerator(AbstractFeatureGenerator):
    def get_feature_names(self) -> list[str]:
        return [
            "commits_x_growth", "interval_x_entropy", "contrib_x_entropy", "average_growth_commit",
            "committer_x_interval_entropy"
        ]

    def generate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df["commits_x_growth"] = df["total_commits"] * df["recent_growth_ratio"]
        df["interval_x_entropy"] = df["avg_commit_interval"] * df["interval_entropy"]
        df["contrib_x_entropy"] = df["recent_contribution_ratio"] * df["interval_entropy"]
        df["average_growth_commit"] = df["cumulative_size"] / df["total_commits"]
        df["committer_x_interval_entropy"] = df["committer_entropy"] * df["interval_entropy"]

        return df
