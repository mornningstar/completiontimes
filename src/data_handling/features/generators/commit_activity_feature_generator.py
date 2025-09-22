import numpy as np
import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator

@feature_generator_registry.register
class CommitActivityFeatureGenerator(AbstractFeatureGenerator):
    def get_feature_names(self) -> list[str]:
        return [
            'commit_num', 'total_commits', 'commits_last_30d', 'commits_last_90d', 'commits_ratio_30d',
            'commits_ratio_90d', 'recent_commit_activity_surge', 'days_since_last_commit', 'is_first_commit',
            'std_commit_interval', 'avg_commit_interval', 'last_3_mean', 'last_3_slope', 'last_5_slope',
            'growth_acceleration', 'days_with_commits_ratio'
        ]

    def generate(self, df: pd.DataFrame, windows: list[int], **kwargs) -> pd.DataFrame:
        df_sorted = df.sort_values(['path', 'date']).copy()
        df_sorted['commit_num'] = df_sorted.groupby('path').cumcount() + 1
        df_sorted['total_commits'] = df_sorted.groupby('path')['commit_num'].transform('max')

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

        window_commit_counts = df_sorted.groupby('path', group_keys=False).apply(
            _calculate_commits_in_windows
        )
        df_sorted = pd.concat([df_sorted, window_commit_counts], axis=1)

        for window in windows:
            df_sorted[f"commits_ratio_{window}d"] = (df_sorted[f"commits_last_{window}d"] / df_sorted["total_commits"]).fillna(0).clip(0, 1)

        df_sorted["recent_commit_activity_surge"] = (
                df_sorted[f"commits_ratio_{windows[0]}d"] - df_sorted[f"commits_ratio_{windows[1]}d"]
        )

        df_sorted["days_since_last_commit"] = df_sorted.groupby("path")["date"].diff().dt.days
        df_sorted["is_first_commit"] = (df_sorted["days_since_last_commit"].isna()).astype(int)

        df_sorted["days_since_last_commit"] = df_sorted["days_since_last_commit"].fillna(0)

        df_sorted["std_commit_interval"] = (
            df_sorted.groupby("path")["days_since_last_commit"].expanding().std().reset_index(level=0, drop=True).fillna(0)
        )

        df_sorted["avg_commit_interval"] = df_sorted.groupby("path")["days_since_last_commit"].transform("mean").fillna(0)


        df_sorted.replace([np.inf, -np.inf], 0, inplace=True)

        first = df_sorted.groupby("path")["date"].transform("min")
        last = df_sorted.groupby("path")["date"].transform("max")
        span_days = (last - first).dt.days + 1

        active_day_counts = df_sorted.groupby("path")["date"].transform("nunique")
        df_sorted["days_with_commits_ratio"] = (active_day_counts / span_days).clip(0, 1).fillna(0)

        all_new_cols = self.get_feature_names()
        cols_to_merge = [col for col in all_new_cols if col in df_sorted.columns]

        output_df = pd.merge(df, df_sorted[['path', 'date'] + cols_to_merge], on=['path', 'date'], how='left')

        return output_df
