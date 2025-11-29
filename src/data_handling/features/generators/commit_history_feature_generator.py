import numpy as np
import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator


@feature_generator_registry.register
class CommitHistoryFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self):
        super().__init__()
        self.windows = None

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        return [
            'commit_num', 'commit_interval_days', 'is_first_commit', 'age_in_days', 'commits_per_day_so_far',
            'std_commit_interval', 'avg_commit_interval', 'weekday', 'month', 'max_commit_interval_to_date'
        ] + [f'commits_last_{w}d' for w in self.windows] + [f'commits_ratio_{w}d' for w in self.windows]

    def generate(self, df: pd.DataFrame, windows=None, **kwargs) -> tuple[pd.DataFrame, list[str]]:
        if windows is None:
            windows = [7, 30]
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

        df.sort_values(['path', 'date'], inplace=True)

        # Basic commit counts
        df['commit_num'] = df.groupby('path').cumcount() + 1

        # Time-windowed commit counts
        window_counts = df.groupby('path', group_keys=False).apply(_calculate_commits_in_windows)
        df = pd.concat([df, window_counts], axis=1)

        for window in windows:
            col_name = f"commits_last_{window}d"
            df[f"commits_ratio_{window}d"] = (df[col_name] / df["commit_num"]).fillna(0)

        if len(windows) >= 2:
            df["recent_commit_activity_surge"] = (
                df[f"commits_ratio_{windows[0]}d"] - df[f"commits_ratio_{self.windows[1]}d"]
            )

        # Commit interval features
        time_diff = df.groupby("path")["date"].diff()
        df["commit_interval_days"] = time_diff.dt.days.fillna(0)
        df['max_commit_interval_to_date'] = (df.groupby('path')['commit_interval_days'].expanding().max().
                                             reset_index(level=0, drop=True))
        df["is_first_commit"] = (df["commit_interval_days"] == 0).astype(int)

        # Expanding statistics on commit intervals
        expanding_std = df.groupby("path")["commit_interval_days"].expanding().std()
        df["std_commit_interval"] = expanding_std.reset_index(level=0, drop=True).fillna(0)
        df["avg_commit_interval"] = (df.groupby("path")["commit_interval_days"].expanding().mean().
                                     reset_index(level=0, drop=True).
                                     fillna(0))

        # Date-based features
        first_commit = df.groupby("path")["date"].transform("min")
        df["age_in_days"] = (df["date"] - first_commit).dt.days

        df["commits_per_day_so_far"] = df["commit_num"] / (df["age_in_days"] + 1)

        df["weekday"] = df["date"].dt.weekday
        df["month"] = df["date"].dt.month

        return df, ['is_first_commit']
