import numpy as np
import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator

@feature_generator_registry.register
class ChangeQualityFeatureGenerator(AbstractFeatureGenerator):
    def get_feature_names(self) -> list[str]:
        return [
            'add_ratio', 'pure_addition', 'pure_deletion', 'pure_addition_count', 'pure_deletion_count'
        ]

    def generate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df["add_ratio"] = (
                df["lines_added"] / df["line_change"].replace(0, np.nan)
        ).fillna(0).clip(0, 1)

        df["pure_addition"] = ((df["lines_added"] > 0) & (df["lines_deleted"] == 0)).astype(int)
        df["pure_deletion"] = ((df["lines_deleted"] > 0) & (df["lines_added"] == 0)).astype(int)

        df["pure_addition_count"] = df.groupby("path")["pure_addition"].cumsum()
        df["pure_deletion_count"] = df.groupby("path")["pure_deletion"].cumsum()

        return df
