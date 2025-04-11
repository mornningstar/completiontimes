import logging

import numpy as np
import pandas as pd

from src.data_handling.database.async_database import AsyncDatabase
from src.visualisations.plotting import Plotter

class FileFeatureEngineer:

    def __init__(self, api_connection, project_name, threshold, consecutive_days):
        self.api_connection = api_connection
        self.project_name = project_name
        self.threshold = threshold
        self.consecutive_days = consecutive_days

        self.plotter = Plotter(self.project_name)
        self.logging = logging.getLogger(self.__class__.__name__)

    async def fetch_all_files(self):

        all_files_data = await AsyncDatabase.fetch_all(
            self.api_connection.file_tracking_collection
        )

        rows = []
        for file_data in all_files_data:
            file_path = file_data['path']
            for commit in file_data.get('commit_history', []):
                rows.append({
                    "path": file_path,
                    "date": pd.to_datetime(commit['date']),
                    "size": commit['size']
                })

        return pd.DataFrame(rows).sort_values('date')

    def calculate_metrics(self, file_df, window=7):
        """
                Calculate rolling and cumulative metrics for files.
        """
        file_df["size"] = pd.to_numeric(file_df["size"], errors="coerce")

        file_df.dropna(subset=["size"], inplace=True)  # Drop rows where size is missing

        file_df.sort_values(["path", "date"], inplace=True)

        file_df["size_diff"] = file_df.groupby("path")["size"].diff().fillna(0).abs()

        roll = file_df.groupby("path")["size"].rolling(window=window)

        file_df[f"rolling_{window}_mean"] = roll.mean().reset_index(level=0, drop=True)
        file_df[f"rolling_{window}_std"] = roll.std().reset_index(level=0, drop=True)
        file_df[f"rolling_{window}_max"] = roll.max().reset_index(level=0, drop=True)
        file_df[f"rolling_{window}_min"] = roll.min().reset_index(level=0, drop=True)
        file_df[f"rolling_{window}_median"] = roll.median().reset_index(level=0, drop=True)
        file_df[f"rolling_{window}_var"] = roll.var().reset_index(level=0, drop=True)

        file_df[f"ema_{window}"] = (
            file_df.groupby("path")["size"]
            .transform(lambda x: x.ewm(span=window, adjust=False).mean())
        )

        # Cumulative Features
        file_df["cumulative_size"] = file_df.groupby("path")["size"].cumsum()
        file_df["cumulative_mean"] = file_df.groupby("path")["size"].expanding().mean().reset_index(level=0, drop=True)
        file_df["cumulative_std"] = file_df.groupby("path")["size"].expanding().std().reset_index(level=0, drop=True)

        # Lag Features
        for lag in range(1, window + 1):
            file_df[f"lag_{lag}_size"] = file_df.groupby("path")["size"].shift(lag)

        # Derived Features
        file_df["absolute_change"] = file_df.groupby("path")["size"].diff().abs()
        file_df["percentage_change"] = (
            file_df.groupby("path")["size"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
        )
        file_df["percentage_change"] = file_df["percentage_change"].round(4)

        file_df[f"rolling_{window}_mean_to_std_ratio"] = (
                file_df[f"rolling_{window}_mean"] / file_df[f"rolling_{window}_std"]
        )

        file_df["days_since_last_commit"] = file_df.groupby("path")["date"].diff().dt.days.fillna(0)
        file_df["commits_last_30d"] = file_df.groupby("path").apply(lambda g: self.count_recent_commits(g)).reset_index(level=0,
                                                                                                         drop=True)

        file_df['last_3_mean'] = file_df[["lag_1_size", "lag_2_size", "lag_3_size"]].mean(axis=1)
        file_df['last_3_slope'] = file_df["lag_1_size"] - file_df["lag_3_size"]

        file_df = file_df.replace([np.inf, -np.inf], 0)

        file_df = self.add_completion_labels(file_df, self.threshold, self.consecutive_days)
        file_df = self.add_days_until_completion(file_df)

        return file_df

    def count_recent_commits(self, group, window_days=30):
        dates = group["date"]
        return dates.apply(lambda d: (dates < d) & (dates >= d - pd.Timedelta(days=window_days))).sum(axis=0)

    def add_completion_labels(self, df, threshold, consecutive_days, idle_days_cutoff=180):
        """
            Add a 'completion_date' column for each file based on two strategies:
            1. A stable pattern: percentage_change stays below threshold for consecutive_days commits
            2. A long period of inactivity after the last commit (idle_days_cutoff)
        """
        df['completion_date'] = pd.NaT
        df['completion_reason'] = None

        now = pd.Timestamp.now().normalize()

        for path, group in df.groupby("path"):
            group = group.copy().sort_values("date")

            # Strategy 1: Detect consecutive low-change commits
            changes = group["percentage_change"].abs().fillna(np.inf)
            below_threshold = changes < threshold

            rolling_sum = below_threshold.rolling(window=consecutive_days, min_periods=consecutive_days).sum()
            valid_indices = rolling_sum[rolling_sum >= consecutive_days].index

            if not valid_indices.empty:
                # Find the corresponding date in the original group
                completion_idx = valid_indices[0]
                raw_date = group.loc[completion_idx, 'date']
                completion_date = pd.Timestamp(raw_date).tz_localize(None).to_pydatetime()
                df.loc[df["path"] == path, "completion_date"] = completion_date
                df.loc[df["path"] == path, "completion_reason"] = "stable_pattern"
                continue

            # Strategy 2: Inactivity fallback
            last_commit_date = group["date"].max()
            last_commit_date = last_commit_date.tz_localize(None).to_pydatetime()
            days_since_last_commit = (now - last_commit_date).days

            if days_since_last_commit > idle_days_cutoff:
                df.loc[df["path"] == path, "completion_date"] = last_commit_date
                df.loc[df["path"] == path, "completion_reason"] = "idle_timeout"

        num_completed_files = df[df['completion_date'].notna()]['path'].nunique()
        total_files = df['path'].nunique()
        self.logging.info(
            f"Completed files: {num_completed_files} / {total_files} ({(num_completed_files / total_files * 100):.2f}%)")

        self.plotter.plot_completion_donut(num_completed_files, total_files)

        return df

    def add_days_until_completion(self, df):
        df = df.copy()

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        df["completion_date"] = pd.to_datetime(df["completion_date"])
        df.loc[df["completion_date"].notnull(), "completion_date"] = (
            df.loc[df["completion_date"].notnull(), "completion_date"].dt.tz_localize(None)
        )

        #df["file_completion_date"] = df.groupby("path")["completion_date"].transform("first")

        df["days_until_completion"] = (
                df["completion_date"] - df["date"]
        ).dt.days

        df["days_until_completion"] = df["days_until_completion"].clip(lower=0)

        return df

    async def save_features_to_db(self, file_features):
        """
        Save the computed features back to the database.
        """
        file_features['completion_date'] = file_features['completion_date'].astype(object).where(
            file_features['completion_date'].notnull(), None
        )

        grouped_features = file_features.groupby("path")

        for path, group in grouped_features:
            features = group.reset_index().to_dict(orient="records")
            query = {"path": path}
            update = {"$set": {"features": features}}  # Save features as a list under "features"
            await AsyncDatabase.update_one(self.api_connection.file_tracking_collection, query, update)

    async def run(self):
        """
        Fetch all files, compute features, and save them back to the database.
        """
        self.logging.info(f"Running feature engineering for project: {self.project_name}")

        file_df = await self.fetch_all_files()
        file_features = self.calculate_metrics(file_df)
        await self.save_features_to_db(file_features)

        self.logging.info(f"Finished feature engineering for project: {self.project_name}")

        return file_features