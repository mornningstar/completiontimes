import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator

@feature_generator_registry.register
class CommitterFeatureGenerator(AbstractFeatureGenerator):
    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        significant_committers = self._get_significant_committers(df)
        if significant_committers.empty:
            return []
        feature_names = [f"committer_{name}" for name in significant_committers]
        feature_names.append("committer_other")
        return feature_names

    def _get_significant_committers(self, df: pd.DataFrame):
        if 'committer' not in df.columns:
            return pd.Index([])
        commit_counts = df['committer'].value_counts()
        total_commits = len(df)
        return commit_counts[commit_counts / total_commits >= 0.01].index

    def generate(self, df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, list[str]]:
        significant_committers = self._get_significant_committers(df)
        if significant_committers.empty:
            return df, []

        df['committer_grouped'] = df['committer'].apply(
            lambda x: x if x in significant_committers else 'other'
        )

        dummies = pd.get_dummies(df['committer_grouped'], prefix='committer')
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=['committer', 'committer_grouped'], inplace=True, errors='ignore')

        return df, dummies.columns.tolist()
