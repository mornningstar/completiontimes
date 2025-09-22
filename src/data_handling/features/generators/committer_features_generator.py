import logging

import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator

@feature_generator_registry.register
class CommitterFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self):
        self.logging = logging.getLogger(self.__class__.__name__)

    def get_feature_names(self) -> list[str]:
        return [
            "committer_"
        ]

    def generate(self, df: pd.DataFrame, use_categorical: bool = False, min_percentage: float = 0.01, **kwargs) -> pd.DataFrame:
        total_commits = len(df)
        committer_counts = df["committer"].value_counts()
        threshold = total_commits * min_percentage
        frequent_committers = committer_counts[committer_counts >= threshold].index

        self.logging.info(f"There are {len(frequent_committers)} frequent committers out of a total of "
                          f"{len(committer_counts)}")

        df["committer_grouped"] = df["committer"].apply(lambda x: x if x in frequent_committers else "other")

        if not use_categorical:
            committer_dummies = pd.get_dummies(df["committer_grouped"], prefix="committer")
            df = pd.concat([df, committer_dummies], axis=1)

        return df
