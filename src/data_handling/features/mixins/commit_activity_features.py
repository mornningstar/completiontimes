import numpy as np
import pandas as pd


class CommitActivityFeatureMixin:
    def _add_commit_activity_features(self, df, windows: list[int]):
        df["total_commits"] = df.groupby("path").cumcount() + 1

        for window in windows:
            df[f"commits_last_{window}d"] = df.groupby("path").apply(
                lambda g: self.count_recent_commits(g, window_days=window)
            ).reset_index(level=0, drop=True)
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


    def count_recent_commits(self, group, window_days=30):
        dates = group["date"]
        result = []
        for i, current_date in enumerate(dates):
            window_start = current_date - pd.Timedelta(days=window_days)
            count = dates.iloc[:i].between(window_start, current_date).sum()
            result.append(count)

        return pd.Series(result, index=group.index)