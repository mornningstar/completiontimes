import numpy as np
import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator


@feature_generator_registry.register
class CommitHistoryFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self):
        super().__init__()
        self.windows = None

    def get_feature_names(self) -> list[str]:
        return [
            'commit_num', 'total_commits', 'days_since_last_commit', 'is_first_commit',
            'std_commit_interval', 'avg_commit_interval', 'weekday', 'month'
        ] + [f'commits_last_{w}d' for w in self.windows] + [f'commits_ratio_{w}d' for w in self.windows]

    def generate(self, df: pd.DataFrame, windows: list[int] = [7, 30], **kwargs) -> pd.DataFrame:
        self.windows = windows

        def _calculate_commits_in_windows(group: pd.DataFrame) -> pd.DataFrame:
            """
            For a single path group, calculate the number of commits in each window.
            """
            dates = group['date']
            commit_nums = group['commit_num']

            result_df = pd.DataFrame(index=group.index)

            for window in windows:
                start_dates = dates - pd.Timedelta(days=window)
                past_indices = dates.searchsorted(start_dates, side='right') - 1

                past_commit_nums = np.where(
                    past_indices >= 0,
                    commit_nums.iloc[past_indices].values,
                    0
                )

                result_df[f'commits_last_{window}d'] = commit_nums.values - past_commit_nums

            return result_df

        df_sorted = df[['path', 'date']].copy()
        df_sorted.sort_values(['path', 'date'], inplace=True)

        # Basic commit counts
        df_sorted['commit_num'] = df_sorted.groupby('path').cumcount() + 1
        df_sorted['total_commits'] = df_sorted.groupby('path')['commit_num'].transform('max')

        # Time-windowed commit counts
        window_counts = df_sorted.groupby('path', group_keys=False).apply(_calculate_commits_in_windows)
        df_sorted = pd.concat([df_sorted, window_counts], axis=1)

        for window in windows:
            col_name = f"commits_last_{window}d"
            df_sorted[f"commits_ratio_{window}d"] = (df_sorted[col_name] / df_sorted["total_commits"]).fillna(0)

        if len(windows) >= 2:
            df_sorted["recent_commit_activity_surge"] = (
                df_sorted[f"commits_ratio_{windows[0]}d"] - df_sorted[f"commits_ratio_{self.windows[1]}d"]
            )

        # Commit interval features
        time_diff = df_sorted.groupby("path")["date"].diff()
        df_sorted["days_since_last_commit"] = time_diff.dt.days.fillna(0)
        df_sorted["is_first_commit"] = df_sorted["days_since_last_commit"].isna().astype(int)

        # Expanding statistics on commit intervals
        expanding_std = df_sorted.groupby("path")["days_since_last_commit"].expanding().std()
        df_sorted["std_commit_interval"] = expanding_std.reset_index(level=0, drop=True).fillna(0)
        df_sorted["avg_commit_interval"] = df_sorted.groupby("path")["days_since_last_commit"].transform("mean").fillna(
            0)

        # Date-based features
        df_sorted["weekday"] = df_sorted["date"].dt.weekday
        df_sorted["month"] = df_sorted["date"].dt.month

        return df_sorted