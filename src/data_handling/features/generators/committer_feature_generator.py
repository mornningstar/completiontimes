import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator

@feature_generator_registry.register
class CommitterFeatureGenerator(AbstractFeatureGenerator):
    def get_feature_names(self) -> list[str]:
        return [
            'committer_grouped'
        ]

    def generate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if 'committer' not in df.columns:
            return df

        commit_counts = df['committer'].value_counts()
        total_commits = len(df)
        significant_committers = commit_counts[commit_counts / total_commits >= 0.01].index

        df['committer_grouped'] = df['committer'].apply(
            lambda x: x if x in significant_committers else 'other'
        )

        return df

