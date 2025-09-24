import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator

@feature_generator_registry.register
class CommitterFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self):
        super().__init__()
        self.feature_names_ = None

    def get_feature_names(self) -> list[str]:
        if self.feature_names_ is None:
            raise RuntimeError(
                "The 'generate' method must be called before 'get_feature_names' "
                "to determine the dynamic feature names."
            )
        return self.feature_names_

    def generate(self, df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, list[str]]:
        if 'committer' not in df.columns:
            return df

        initial_columns = set(df.columns)

        commit_counts = df['committer'].value_counts()
        total_commits = len(df)
        significant_committers = commit_counts[commit_counts / total_commits >= 0.01].index

        df['committer_grouped'] = df['committer'].apply(
            lambda x: x if x in significant_committers else 'other'
        )

        dummies = pd.get_dummies(df['committer_grouped'], prefix='committer')
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=['committer', 'committer_grouped'], inplace=True, errors='ignore')

        final_columns = set(df.columns)
        self.feature_names_ = list(final_columns - initial_columns)

        return df, self.feature_names_
