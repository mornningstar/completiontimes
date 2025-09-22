import numpy as np
import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator

@feature_generator_registry.register
class TimeSeriesFeatureGenerator(AbstractFeatureGenerator):
    def get_feature_names(self) -> list[str]:
        return [
            "size_diff", "std_dev_size_diff", "rolling_7_mean", "rolling_7_std", "rolling_7_max", "rolling_7_min",
            "rolling_7_median", "rolling_7_var", "ema_7", "cumulative_size", "cum_lines_added", "cum_lines_deleted",
            "cum_line_change", "cumulative_mean", "cumulative_std", "lag_1_size", "lag_2_size", "lag_3_size",
            "lag_4_size", "lag_5_size", "lag_6_size", "lag_7_size", "recent_growth_ratio", "absolute_change",
            "percentage_change", "rolling_7_mean_to_std_ratio"
        ]

    def generate(self, df: pd.DataFrame, window: int = 7, n: int = 5, **kwargs) -> pd.DataFrame:
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

        df["last_3_mean"] = df[["lag_1_size", "lag_2_size", "lag_3_size"]].mean(axis=1)
        df["last_3_slope"] = df["lag_1_size"] - df["lag_3_size"]
        df["last_5_slope"] = df["lag_1_size"] - df["lag_5_size"]
        df["growth_acceleration"] = df["last_3_slope"] - df["last_5_slope"]

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
