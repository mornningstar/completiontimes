import logging

import numpy as np
import pandas as pd

from src.data_handling.database.async_database import AsyncDatabase
from src.data_handling.database.file_repo import FileRepository
from src.visualisations.model_plotting import ModelPlotter


class FileFeatureEngineer:

    def __init__(self, file_repo: FileRepository, plotter: ModelPlotter, threshold, consecutive_days):
        self.file_repo = file_repo
        self.threshold = threshold
        self.consecutive_days = consecutive_days

        self.plotter = plotter
        self.logging = logging.getLogger(self.__class__.__name__)

    async def fetch_all_files(self):

        all_files_data = await self.file_repo.get_all()

        rows = []
        for file_data in all_files_data:
            file_path = file_data['path']
            for commit in file_data.get('commit_history', []):
                rows.append({
                    "path": file_path,
                    "date": pd.to_datetime(commit['date']),
                    "size": commit['size'],
                    "committer": commit['committer']
                })

        return pd.DataFrame(rows).sort_values('date')

    def _add_metadata_features(self, df):
        """
        All the features that are extracted from file paths, directoy structures, etc.
        :param df:
        :return:
        """
        df["file_extension"] = df["path"].str.extract(r'\.([a-zA-Z0-9]+)$')[0].fillna("unknown")
        ext_counts = df["file_extension"].value_counts()
        top_exts = ext_counts[(ext_counts >= 10) | (ext_counts.rank(method="min") <= 10)].index
        df["file_extension"] = df["file_extension"].apply(lambda x: x if x in top_exts else "other")
        dummies = pd.get_dummies(df["file_extension"], prefix="ext")
        df = pd.concat([df, dummies], axis=1)

        df["path_depth"] = df["path"].apply(lambda x: x.count("/"))
        df["in_test_dir"] = df["path"].str.lower().str.contains(r"/tests?/").astype(int)
        df["in_docs_dir"] = df["path"].str.lower().str.contains(r"/(?:docs|documentation)/").astype(int)
        df["weekday"] = df["date"].dt.weekday  # Monday = 0
        df["month"] = df["date"].dt.month  # January = 1
        first_commit = df.groupby("path")["date"].transform("min")
        df["age_in_days"] = (df["date"] - first_commit).dt.days

        path_lower = df["path"].str.lower()
        config_extensions = {"json", "yaml", "yml", "ini", "toml", "env", "cfg", "conf", "ini"}
        source_code_exts = (".py", ".js", ".ts", ".rb", ".java", ".cpp", ".c", ".cs", ".go", ".rs", ".php", ".swift")

        df["is_config_file"] = df["file_extension"].isin(config_extensions)
        df["is_markdown"] = path_lower.str.endswith((".md", ".markdown")).astype(int)
        df["is_desktop_entry"] = path_lower.str.endswith(".desktop").astype(int)
        df["is_workflow_file"] = path_lower.str.contains(r"\.github/workflows/").astype(int)
        df["has_readme_name"] = path_lower.str.contains(r"readme").astype(int)
        df["is_source_code"] = path_lower.str.endswith(source_code_exts).astype(int)
        df["is_script"] = path_lower.str.endswith((".sh", ".bat")).astype(int)

        return df

    def _add_time_series_stats(self, df, window=7, n=5):
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
        df.dropna(subset=["size"], inplace=True)
        df["size_diff"] = df.groupby("path")["size"].diff().fillna(0).abs()
        df.sort_values(["path", "date"], inplace=True)

        df["std_dev_size_diff"] = df.groupby("path")["size_diff"].transform("std").fillna(0)

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
        df["avg_commit_interval"] = df.groupby("path")["days_since_last_commit"].transform("mean").fillna(0)
        df["last_3_mean"] = df[["lag_1_size", "lag_2_size", "lag_3_size"]].mean(axis=1)
        df["last_3_slope"] = df["lag_1_size"] - df["lag_3_size"]
        df.replace([np.inf, -np.inf], 0, inplace=True)

        first = df.groupby("path")["date"].transform("min")
        last = df.groupby("path")["date"].transform("max")
        span_days = (last - first).dt.days + 1
        active_day_counts = df.groupby("path")["date"].nunique()
        df = df.merge(active_day_counts.rename("active_days"), on="path", how="left")
        df["days_with_commits_ratio"] = (df["active_days"] / span_days).clip(0, 1).fillna(0)
        df.drop(columns=["active_days"], inplace=True)

        return df

    def _add_temporal_dynamics_features(self, df):
        df["commit_interval_days"] = df.groupby("path")["date"].diff().dt.days.fillna(0)

        def entropy(series):
            counts = series.value_counts(normalize=True)
            return -np.sum(counts * np.log2(counts + 1e-10)) # 1e-10 for numerical stability

        df["interval_entropy"] = df.groupby("path")["commit_interval_days"].transform(entropy)

        def early_growth_ratio(group, n=7):
            sorted_group = group.sort_values("date")
            size_diffs = sorted_group["size"].diff().fillna(0)

            n = min(n, len(size_diffs))
            early_sum = size_diffs.iloc[:n].sum()
            total_growth = size_diffs.sum()
            return early_sum / (total_growth + 1e-10)

        def compute_ratio(group, n=5):
            size_diff = group["size"].diff().fillna(0).abs()
            total_growth = size_diff.cumsum()

            recent_sum = size_diff.rolling(window=n, min_periods=1).sum()
            ratio = (recent_sum / (total_growth + 1e-10)).clip(0, 1)

            return ratio

        df["normalized_early_growth"] = (
            df.groupby("path").apply(early_growth_ratio)
            .reset_index(level=0, drop=True)
            #.reindex(df["path"].values)
            #.values
        )

        df["recent_contribution_ratio"] = (
            df.groupby("path").apply(compute_ratio).reset_index(level=0, drop=True)
        )

        return df

    def count_recent_commits(self, group, window_days=30):
        dates = group["date"]
        result = []
        for i, current_date in enumerate(dates):
            window_start = current_date - pd.Timedelta(days=window_days)
            # Nur die bisherigen Commits zählen (inkl. aktueller Zeile, wenn gewünscht)
            count = dates.iloc[:i].between(window_start, current_date).sum()
            result.append(count)
        return pd.Series(result, index=group.index)

    def _add_feature_interactions(self, df):
        df["commits_x_growth"] = df["total_commits"] * df["recent_growth_ratio"]
        df["interval_x_entropy"] = df["avg_commit_interval"] * df["interval_entropy"]
        df["contrib_x_entropy"] = df["recent_contribution_ratio"] * df["interval_entropy"]
        df["growth_x_age"] = df["recent_growth_ratio"] * df["age_in_days"]
        df["average_growth_commit"] = df["cumulative_size"] / df["total_commits"]

        return df

    def _add_committer_features(self, df, min_percentage=0.01):
        total_commits = len(df)
        committer_counts = df["committer"].value_counts()
        threshold = total_commits * min_percentage
        frequent_committers = committer_counts[committer_counts >= threshold].index

        self.logging.info(f"There are {len(frequent_committers)} frequent committers out of a total of "
                          f"{len(committer_counts)}")

        df["committer_grouped"] = df["committer"].apply(lambda x: x if x in frequent_committers else "other")
        committer_dummies = pd.get_dummies(df["committer_grouped"], prefix="committer")
        df = pd.concat([df, committer_dummies], axis=1)

        return df

    def calculate_metrics(self, file_df, window=7):
        """
        Calculate additional metrics for files.
        :param file_df: the file dataset with its commits per file
        :param window: the defined window for lag and ema features
        :return:
        """
        # Use File Extensions as Features
        file_df = self._add_metadata_features(file_df)
        file_df = self._add_time_series_stats(file_df, window=window)
        file_df = self._add_commit_activity_features(file_df, windows=[30, 90])
        file_df = self._add_temporal_dynamics_features(file_df)
        file_df = self._add_feature_interactions(file_df)
        file_df = self._add_committer_features(file_df)

        file_df = self.add_completion_labels(file_df, self.threshold, self.consecutive_days)
        file_df = self.add_days_until_completion(file_df)

        numeric_cols = [col for col in file_df.select_dtypes(include=[np.number]).columns
                        if col != "days_until_completion"]
        file_df[numeric_cols] = file_df[numeric_cols].fillna(0.0)

        return file_df

    def add_completion_labels(self, df, threshold, consecutive_days, idle_days_cutoff=180):
        """
            Add a 'completion_date' column for each file based on two strategies:
            1. A stable pattern: percentage_change stays below threshold for consecutive_days commits
            2. A deletion event:
            3. A long period of inactivity after the last commit (idle_days_cutoff)
        """
        df['completion_date'] = pd.NaT
        df['completion_reason'] = None

        now = pd.Timestamp.now().normalize()

        for path, group in df.groupby("path"):
            group = group.copy().sort_values("date")

            # Strategy 1: Detect consecutive low-change commits
            pct = group["percentage_change"].abs().fillna(0)
            sum_pct = pct.rolling(window=consecutive_days,
                                  min_periods=consecutive_days).sum()

            valid = sum_pct[sum_pct < threshold]

            if not valid.empty:
                # Find the corresponding date in the original group
                completion_idx = valid.index[-1]
                raw_date = group.loc[completion_idx, 'date']
                completion_date = pd.Timestamp(raw_date).tz_localize(None).to_pydatetime()

                df.loc[df["path"] == path, "completion_date"] = completion_date
                df.loc[df["path"] == path, "completion_reason"] = "stable_pattern"
                continue

            # Strategy 2: Explicit deletion (size = 0)
            if group["size"].iloc[-1] == 0:
                deletion_date = group["date"].iloc[-1]
                deletion_date = pd.to_datetime(deletion_date).tz_localize(None)
                df.loc[df["path"] == path, "completion_date"] = deletion_date
                df.loc[df["path"] == path, "completion_reason"] = "deleted"
                continue

            # Strategy 3: Inactivity fallback
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

        df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce").dt.tz_localize(None)
        df.loc[df["completion_date"].notnull(), "completion_date"] = (
            df.loc[df["completion_date"].notnull(), "completion_date"].dt.tz_localize(None)
        )

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
            await self.file_repo.append_features_to_file(path, features, upsert=False)

    async def run(self):
        """
        Fetch all files, compute features, and save them back to the database.
        """

        file_df = await self.fetch_all_files()
        file_features = self.calculate_metrics(file_df)

        feature_cols = [col for col in file_features.select_dtypes(include="number").columns
                        if col not in ["days_until_completion", "size", "cumulative_size"]]
        target_series = file_features["days_until_completion"]
        self.plotter.plot_feature_correlations(file_features[feature_cols], target_series)

        await self.save_features_to_db(file_features)



        return file_features