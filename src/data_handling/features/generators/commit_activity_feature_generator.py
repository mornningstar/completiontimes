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
        df_sorted = df.sort_values('date').copy()
        df_sorted['commit_num'] = df_sorted.groupby('path').cumcount() + 1
        df_sorted['total_commits'] = df_sorted.groupby('path')['commit_num'].transform('max')

        # --- Efficient Rolling Commit Counts using merge_asof ---
        for window in windows:
            df_sorted[f'window_start_date_{window}'] = df_sorted['date'] - pd.Timedelta(days=window)

            # merge_asof finds the last commit before the window_start_date
            merged = pd.merge_asof(
                df_sorted[['path', 'date', 'window_start_date_{}'.format(window), 'commit_num']],
                df_sorted[['path', 'date', 'commit_num']],
                on='date',
                by='path',
                left_on='window_start_date_{}'.format(window),
                direction='backward'
            )
            # The difference in commit numbers gives the count in the window
            df_sorted[f'commits_last_{window}d'] = (merged['commit_num_x'] - merged['commit_num_y']).fillna(
                merged['commit_num_x'])

        # Now, merge these calculated features back to the original dataframe
        cols_to_merge = ['path', 'date', 'total_commits'] + [f'commits_last_{w}d' for w in windows]
        df = pd.merge(df, df_sorted[cols_to_merge], on=['path', 'date'], how='left')
        # --- End of efficient calculation ---

        for window in windows:
            df[f"commits_ratio_{window}d"] = (df[f"commits_last_{window}d"] / df["total_commits"]).fillna(0).clip(0, 1)
            df[f"commits_ratio_{window}d_smooth"] = (
                df.groupby("path")[f"commits_ratio_{window}d"]
                .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            )

        df["recent_commit_activity_surge"] = (
                df["commits_ratio_30d"] - df["commits_ratio_90d"]
        )

        df["days_since_last_commit"] = df.groupby("path")["date"].diff().dt.days.fillna(0)
        df["is_first_commit"] = (df["days_since_last_commit"] == 0).astype(int)

        median = df["days_since_last_commit"].median()
        df["days_since_last_commit"] = df["days_since_last_commit"].fillna(median)

        df["std_commit_interval"] = (
            df.groupby("path")["days_since_last_commit"].expanding().std().reset_index(level=0, drop=True).fillna(0)
        )

        df["avg_commit_interval"] = df.groupby("path")["days_since_last_commit"].transform("mean").fillna(0)
        df["last_3_mean"] = df[["lag_1_size", "lag_2_size", "lag_3_size"]].mean(axis=1)
        df["last_3_slope"] = df["lag_1_size"] - df["lag_3_size"]
        df["last_5_slope"] = df["lag_1_size"] - df["lag_5_size"]
        df["growth_acceleration"] = df["last_3_slope"] - df["last_5_slope"]
        df.replace([np.inf, -np.inf], 0, inplace=True)

        first = df.groupby("path")["date"].transform("min")
        last = df.groupby("path")["date"].transform("max")
        span_days = (last - first).dt.days + 1
        active_day_counts = df.groupby("path")["date"].nunique()
        df = df.merge(active_day_counts.rename("active_days"), on="path", how="left")
        df["days_with_commits_ratio"] = (df["active_days"] / span_days).clip(0, 1).fillna(0)
        df.drop(columns=["active_days"], inplace=True)

        return df
