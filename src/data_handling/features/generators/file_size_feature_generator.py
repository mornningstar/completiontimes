import numpy as np
import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator

@feature_generator_registry.register
class FileSizeFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self):
        super().__init__()
        self.window = None
        self.lags = None
        self.recent_n = None
        self.rolling_stats = ['mean', 'std', 'max', 'min', 'median']

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        return [
            'size_diff', 'cumulative_size', 'cumulative_mean_size', 'cumulative_std_size', 'recent_growth_ratio',
            'percentage_change_size'] + [f'rolling_{self.window}_{stat}_size' for stat in self.rolling_stats] + [
            f'ema_{self.window}_size'] + [f'lag_{lag}_size' for lag in range(1, self.lags + 1)]

    def generate(self, df: pd.DataFrame, window: int = 7, lags: int = 5, recent_n: int = 5, **kwargs) -> tuple[pd.DataFrame, list[str]]:
        self.window = window
        self.lags = lags
        self.recent_n = recent_n

        df['size'] = pd.to_numeric(df['size'], errors='coerce')
        df.dropna(subset=['size'], inplace=True)
        df.sort_values(['path', 'date'], inplace=True)

        # Basic size difference
        df['size_diff'] = df.groupby('path')['size'].diff().fillna(0)

        # Cumulative and expanding stats
        df['cumulative_size'] = df.groupby('path')['size'].cumsum()
        df['cumulative_mean_size'] = df.groupby('path')['size'].expanding().mean().reset_index(
            level=0, drop=True)
        df['cumulative_std_size'] = df.groupby('path')['size'].expanding().std().reset_index(
            level=0, drop=True).fillna(0)

        # Rolling window statistics
        for stat in ['mean', 'std', 'max', 'min', 'median']:
            rolled = df.groupby('path')['size'].rolling(window=window, min_periods=1)
            df[f'rolling_{window}_{stat}_size'] = getattr(rolled, stat)().reset_index(level=0, drop=True)

        # Exponential Moving Average
        df[f'ema_{window}_size'] = df.groupby('path')['size'].transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )

        # Lag features
        for lag in range(1, lags + 1):
            df[f'lag_{lag}_size'] = df.groupby('path')['size'].shift(lag)

        # Growth and change metrics
        total_growth_so_far = df['size_diff'].groupby(df['path']).cumsum().replace(0, 1e-9)
        recent_sum = df['size_diff'].groupby(df['path']).transform(
            lambda x: x.rolling(window=recent_n, min_periods=1).sum()
        )

        df['recent_growth_ratio'] = (recent_sum / total_growth_so_far).fillna(0)
        df['percentage_change_size'] = df.groupby('path')['size'].pct_change().fillna(0)

        # Replace infinite values that can arise from pct_change
        df.replace([np.inf, -np.inf], 0, inplace=True)

        return df, []
