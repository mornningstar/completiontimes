import numpy as np
import pandas as pd


class TimeSeriesFeatureMixin:
    def _add_time_series_stats(self, df, window=7, n=5):
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
        df["size_diff"] = df.groupby("path")["size"].diff().fillna(0)
        df.dropna(subset=["size"], inplace=True)
        df.sort_values(["path", "date"], inplace=True)

        df["std_dev_size_diff"] = df.groupby("path")["size_diff"].transform("std").fillna(0)

        for stat in ["mean", "std", "max", "min", "median", "var"]:
            shifted = df.groupby("path")["size"].shift(1)
            rolled = shifted.groupby(df["path"]).rolling(window=window, min_periods=1)
            df[f"rolling_{window}_{stat}"] = getattr(rolled, stat)().reset_index(level=0, drop=True)

        df[f"ema_{window}"] = (
            df.groupby("path")["size"]
            .transform(lambda x: x.ewm(span=window, adjust=False).mean())
        )

        df["cumulative_size"] = df.groupby("path")["size"].shift(1).cumsum()
        df["cum_lines_added"] = df.groupby("path")["lines_added"].cumsum()
        df["cum_lines_deleted"] = df.groupby("path")["lines_deleted"].cumsum()
        df["cum_line_change"] = df["cum_lines_added"] + df["cum_lines_deleted"]
        df["cumulative_mean"] = df.groupby("path")["size"].expanding().mean().reset_index(level=0, drop=True)
        df["cumulative_std"] = df.groupby("path")["size"].expanding().std().reset_index(level=0, drop=True)

        for lag in range(1, window + 1):
            df[f"lag_{lag}_size"] = df.groupby("path")["size"].shift(lag)

        df["recent_growth_ratio"] = df.groupby("path")["size_diff"].transform(
            lambda x: x.rolling(window=n, min_periods=1).sum() / x.cumsum()
        ).fillna(0).clip(0, 1)

        df["absolute_change"] = df.groupby("path")["size"].diff().abs()
        df["percentage_change"] = (
            df.groupby("path")["size"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .round(4)
        )
        df["rolling_7_mean_to_std_ratio"] = (
                df["rolling_7_mean"] / df["rolling_7_std"]
        ).replace([np.inf, -np.inf], 0)

        return df