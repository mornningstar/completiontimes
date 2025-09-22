import numpy as np
import pandas as pd


class DataSplitter:
    @staticmethod
    def split_by_file(file_data_df: pd.DataFrame, test_ratio: float = 0.2, random_state=42):
        file_data_df = file_data_df.copy()
        valid_paths = file_data_df[file_data_df["days_until_completion"].notna()]["path"].unique()

        # Shuffle paths to avoid any ordering bias
        np.random.seed(random_state)
        np.random.shuffle(valid_paths)

        split_idx = int(len(valid_paths) * (1 - test_ratio))
        train_paths = valid_paths[:split_idx]
        test_paths = valid_paths[split_idx:]

        train_df = file_data_df[file_data_df["path"].isin(train_paths)].dropna(subset=["days_until_completion"])
        test_df = file_data_df[file_data_df["path"].isin(test_paths)].dropna(subset=["days_until_completion"])

        return train_df, test_df
