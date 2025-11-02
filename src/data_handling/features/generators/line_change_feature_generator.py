import pandas as pd
from config import globals

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator


@feature_generator_registry.register
class LineChangeFeatureGenerator(AbstractFeatureGenerator):
    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        return [
            'add_ratio', 'pure_addition', 'pure_deletion', 'cum_lines_added', 'cum_lines_deleted', 'cum_line_change',
            'cum_pure_addition', 'cum_pure_deletion', 'consecutive_stable_commits'
        ]

    def generate(self, df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, list[str]]:
        line_change_total = df['lines_added'] + df['lines_deleted']

        # Ratios and purity of changes
        df['add_ratio'] = (df['lines_added'] / line_change_total.replace(0, 1e-9)).fillna(0)
        df['pure_addition'] = ((df['lines_added'] > 0) & (df['lines_deleted'] == 0)).astype(int)
        df['pure_deletion'] = ((df['lines_deleted'] > 0) & (df['lines_added'] == 0)).astype(int)

        # Cumulative counts of change types
        df['cum_lines_added'] = df.groupby('path')['lines_added'].cumsum()
        df['cum_lines_deleted'] = df.groupby('path')['lines_deleted'].cumsum()
        df['cum_line_change'] = df.groupby('path')['line_change'].cumsum()
        df['cum_pure_addition'] = df.groupby('path')['pure_addition'].cumsum()
        df['cum_pure_deletion'] = df.groupby('path')['pure_deletion'].cumsum()

        df['is_stable_commit'] = (df['line_change'] <= globals.STABILITY_ABS_THRESHOLD_LINES).astype(int)

        df['block_id'] = df.groupby('path', group_keys=False)['is_stable_commit'].apply(
            lambda x: (x.shift(1) != x).cumsum()
        ).fillna(1)

        df['consecutive_stable_commits'] = df.groupby(['path', 'block_id'])['is_stable_commit'].cumsum()

        df.drop(
            columns=['is_stable_commit', 'block_id'],
            inplace=True,
            errors='ignore'
        )

        return df, ['pure_addition', 'pure_deletion']
