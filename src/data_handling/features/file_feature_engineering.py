import logging

import numpy as np
import pandas as pd

from src.data_handling.database.async_database import AsyncDatabase
from src.visualisations.model_plotting import ModelPlotter


class FileFeatureEngineer:

    def __init__(self, api_connection, project_name, threshold, consecutive_days, images_dir):
        self.api_connection = api_connection
        self.project_name = project_name
        self.threshold = threshold
        self.consecutive_days = consecutive_days

        self.plotter = ModelPlotter(self.project_name, images_dir=images_dir)
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

    def _add_file_extension_features(self, df):
        df["file_extension"] = df["path"].str.extract(r'\.([a-zA-Z0-9]+)$')[0].fillna("unknown")
        ext_counts = df["file_extension"].value_counts()
        top_exts = ext_counts[(ext_counts >= 10) | (ext_counts.rank(method="min") <= 10)].index
        df["file_extension"] = df["file_extension"].apply(lambda x: x if x in top_exts else "other")
        dummies = pd.get_dummies(df["file_extension"], prefix="ext")
        return pd.concat([df, dummies], axis=1)

    def _add_size_features(self, df, window):
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
        df.dropna(subset=["size"], inplace=True)
        df["size_diff"] = df.groupby("path")["size"].diff().fillna(0).abs()
        df.sort_values(["path", "date"], inplace=True)

        roll = df.groupby("path")["size"].rolling(window=window)
        for stat in ["mean", "std", "max", "min", "median", "var"]:
            df[f"rolling_{window}_{stat}"] = getattr(roll, stat)().reset_index(level=0, drop=True)

        df[f"ema_{window}"] = (
            df.groupby("path")["size"]
            .transform(lambda x: x.ewm(span=window, adjust=False).mean())
        )

        df["cumulative_size"] = df.groupby("path")["size"].cumsum()
        df["cumulative_mean"] = df.groupby("path")["size"].expanding().mean().reset_index(level=0, drop=True)
        df["cumulative_std"] = df.groupby("path")["size"].expanding().std().reset_index(level=0, drop=True)

        return df

    def _add_lag_features(self, df, window):
        for lag in range(1, window + 1):
            df[f"lag_{lag}_size"] = df.groupby("path")["size"].shift(lag)
        return df

    def _add_derived_features(self, df):
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

    def _add_commit_activity_features(self, df):
        df["days_since_last_commit"] = df.groupby("path")["date"].diff().dt.days.fillna(0)
        df["commits_last_30d"] = df.groupby("path").apply(self.count_recent_commits).reset_index(level=0, drop=True)

        df["last_3_mean"] = df[["lag_1_size", "lag_2_size", "lag_3_size"]].mean(axis=1)
        df["last_3_slope"] = df["lag_1_size"] - df["lag_3_size"]

        df.replace([np.inf, -np.inf], 0, inplace=True)
        return df

    def count_recent_commits(self, group, window_days=30):
        dates = group["date"]
        return dates.apply(lambda d: (dates < d) & (dates >= d - pd.Timedelta(days=window_days))).sum(axis=0)

    def calculate_metrics(self, file_df, window=7):
        """
        Calculate additional metrics for files.
        :param file_df: the file dataset with its commits per file
        :param window: the defined window for lag and ema features
        :return:
        """
        # Use File Extensions as Features
        file_df = self._add_file_extension_features(file_df)
        file_df = self._add_size_features(file_df, window)
        file_df = self._add_lag_features(file_df, window)
        file_df = self._add_derived_features(file_df)
        file_df = self._add_commit_activity_features(file_df)

        file_df = self.add_completion_labels(file_df, self.threshold, self.consecutive_days)
        file_df = self.add_days_until_completion(file_df)

        return file_df


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

        feature_cols = [col for col in file_features.select_dtypes(include="number").columns
                        if col not in ["days_until_completion", "size", "cumulative_size"]]
        target_series = file_features["days_until_completion"]
        self.plotter.plot_feature_correlations(file_features[feature_cols], target_series)

        await self.save_features_to_db(file_features)

        self.logging.info(f"Finished feature engineering for project: {self.project_name}")

        return file_features