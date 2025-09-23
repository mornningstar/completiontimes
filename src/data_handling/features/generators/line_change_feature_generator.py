import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator


@feature_generator_registry.register
class LineChangeFeatureGenerator(AbstractFeatureGenerator):
    def get_feature_names(self) -> list[str]:
        return [
            'add_ratio', 'pure_addition', 'pure_deletion', 'cum_lines_added', 'cum_lines_deleted', 'cum_line_change',
            'cum_pure_addition', 'cum_pure_deletion'
        ]

    def generate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_sorted = df[['path', 'date', 'lines_added', 'lines_deleted']].copy()
        df_sorted.sort_values(['path', 'date'], inplace=True)

        line_change_total = df_sorted['lines_added'] + df_sorted['lines_deleted']

        # Ratios and purity of changes
        df_sorted['add_ratio'] = (df_sorted['lines_added'] / line_change_total.replace(0, 1e-9)).fillna(0)
        df_sorted['pure_addition'] = ((df_sorted['lines_added'] > 0) & (df_sorted['lines_deleted'] == 0)).astype(int)
        df_sorted['pure_deletion'] = ((df_sorted['lines_deleted'] > 0) & (df_sorted['lines_added'] == 0)).astype(int)

        # Cumulative counts of change types
        df_sorted['cum_lines_added'] = df_sorted.groupby('path')['lines_added'].cumsum()
        df_sorted['cum_lines_deleted'] = df_sorted.groupby('path')['lines_deleted'].cumsum()
        df_sorted['cum_line_change'] = df_sorted['cum_lines_added'] + df_sorted['cum_lines_deleted']
        df_sorted['cum_pure_addition'] = df_sorted.groupby('path')['pure_addition'].cumsum()
        df_sorted['cum_pure_deletion'] = df_sorted.groupby('path')['pure_deletion'].cumsum()

        return df_sorted
