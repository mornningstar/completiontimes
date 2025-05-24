import logging

import numpy as np
import pandas as pd

from src.data_handling.database.async_database import AsyncDatabase
from src.data_handling.database.file_repo import FileRepository
from src.visualisations.model_plotting import ModelPlotter


class FileFeatureEngineer:

    def __init__(self, file_repo: FileRepository, plotter: ModelPlotter):
        self.file_repo = file_repo
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
                    "committer": commit['committer'],
                    "lines_added": commit.get("additions", 0),
                    "lines_deleted": commit.get("deletions", 0),
                    "line_change": commit.get("total_changes", 0)
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
        config_extensions = {"json", "yaml", "yml", "ini", "toml", "env", "cfg", "conf"}
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
        df["size_diff"] = df.groupby("path")["size"].diff().fillna(0)
        df.dropna(subset=["size"], inplace=True)
        df.sort_values(["path", "date"], inplace=True)

        df["size_diff"] = df.groupby("path")["size"].diff().fillna(0)
        df["std_dev_size_diff"] = df.groupby("path")["size_diff"].transform("std").fillna(0)

        roll = df.groupby("path")["size"].rolling(window=window)
        for stat in ["mean", "std", "max", "min", "median", "var"]:
            df[f"rolling_{window}_{stat}"] = getattr(roll, stat)().reset_index(level=0, drop=True)

        df[f"ema_{window}"] = (
            df.groupby("path")["size"]
            .transform(lambda x: x.ewm(span=window, adjust=False).mean())
        )

        df["cumulative_size"] = df.groupby("path")["size"].cumsum()
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

    def _add_commit_activity_features(self, df, windows: list[int]):
        df["total_commits"] = df.groupby("path").cumcount() + 1
        df["commits_per_day_so_far"] = df["total_commits"] / (df["age_in_days"] + 1)

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
        df["is_first_commit"] = df["days_since_last_commit"].isna().astype(int)
        df["days_since_last_commit"].fillna(df["days_since_last_commit"].median(), inplace=True)

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

    def _add_temporal_dynamics_features(self, df):
        df["commit_interval_days"] = df.groupby("path")["date"].diff().dt.days.fillna(0)

        def entropy(series):
            counts = series.value_counts(normalize=True)
            return -np.sum(counts * np.log2(counts + 1e-10)) # 1e-10 for numerical stability

        df["interval_entropy"] = df.groupby("path")["commit_interval_days"].transform(entropy)
        df["committer_entropy"] = df.groupby("path")["committer"].transform(entropy)

        df["add_entropy"] = df.groupby("path")["lines_added"].transform(entropy)
        df["deletions_entropy"] = df.groupby("path")["lines_deleted"].transform(entropy)

        df = self._add_normalized_early_growth(df, early_n=7)
        df = self._add_recent_contribution_ratio(df, n=5)

        df["commit_interval_std"] = (
            df.groupby("path")["commit_interval_days"].transform("std").fillna(0)
        )

        return df

    def _add_normalized_early_growth(self, df, early_n=7):
        df["early_growth"] = (
            df.groupby("path")["size_diff"]
            .transform(lambda x: x.rolling(window=early_n, min_periods=1).sum())
        )

        df["total_growth_so_far"] = df.groupby("path")["size_diff"].cumsum() + 1e-10  # Avoid div by 0

        df["normalized_early_growth"] = (
                df["early_growth"] / df["total_growth_so_far"]
        ).clip(0, 1)

        return df

    def _add_recent_contribution_ratio(self, df, n=5):
        df["recent_sum"] = (
            df.groupby("path")["size_diff"]
            .transform(lambda x: x.rolling(window=n, min_periods=1).sum())
        )

        df["recent_contribution_ratio"] = (df["recent_sum"] / df["total_growth_so_far"]).clip(0, 1)

        return df

    def _add_change_quality_features(self, df):
        df["add_ratio"] = (
                df["lines_added"] / df["line_change"].replace(0, np.nan)
        ).fillna(0).clip(0, 1)

        df["pure_addition"] = ((df["lines_added"] > 0) & (df["lines_deleted"] == 0)).astype(int)
        df["pure_deletion"] = ((df["lines_deleted"] > 0) & (df["lines_added"] == 0)).astype(int)

        df["pure_addition_count"] = df.groupby("path")["pure_addition"].cumsum()
        df["pure_deletion_count"] = df.groupby("path")["pure_deletion"].cumsum()
        
        return df

    def count_recent_commits(self, group, window_days=30):
        dates = group["date"]
        result = []
        for i, current_date in enumerate(dates):
            window_start = current_date - pd.Timedelta(days=window_days)
            count = dates.iloc[:i].between(window_start, current_date).sum()
            result.append(count)
        return pd.Series(result, index=group.index)

    def _add_feature_interactions(self, df):
        df["commits_x_growth"] = df["total_commits"] * df["recent_growth_ratio"]
        df["interval_x_entropy"] = df["avg_commit_interval"] * df["interval_entropy"]
        df["contrib_x_entropy"] = df["recent_contribution_ratio"] * df["interval_entropy"]
        df["growth_x_age"] = df["recent_growth_ratio"] * df["age_in_days"]
        df["average_growth_commit"] = df["cumulative_size"] / df["total_commits"]
        df["committer_x_interval_entropy"] = df["committer_entropy"] * df["interval_entropy"]

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
        file_df = file_df.groupby("path").filter(lambda g: len(g) >= 5)

        file_df = self._add_metadata_features(file_df)
        file_df = self._add_time_series_stats(file_df, window=window)
        file_df = self._add_commit_activity_features(file_df, windows=[30, 90])
        file_df = self._add_temporal_dynamics_features(file_df)
        file_df = self._add_feature_interactions(file_df)
        file_df = self._add_committer_features(file_df)
        file_df = self._add_change_quality_features(file_df)

        file_df = self.add_completion_labels(file_df)
        file_df = self.add_days_until_completion(file_df)

        numeric_cols = [col for col in file_df.select_dtypes(include=[np.number]).columns
                        if col != "days_until_completion"]
        file_df[numeric_cols] = file_df[numeric_cols].fillna(0.0)

        return file_df

    def _check_stable_line_change_window(self, group):
        group = group.sort_values("date").reset_index(drop=True)
        consecutive_commits = int(len(group) * 0.2)
        commit_window = min(max(3, consecutive_commits), 14)

        latest_valid_completion_date = None
        
        for i in range(len(group)):
            current_line_change = group.loc[i, "line_change"]
            median_change = group.loc[:i, "line_change"].median()
            threshold = max(3, median_change * 0.15)

            if current_line_change <= threshold:
                remaining = len(group) - (i + 1)
                window_size = min(commit_window, remaining)
                if window_size == 0:
                    continue

                next_commits = group.iloc[i + 1: i + 1 + window_size]

                if (next_commits["line_change"] < threshold).all():
                    latest_valid_completion_date = next_commits.iloc[-1]["date"]

        if latest_valid_completion_date:
            return latest_valid_completion_date, "stable_line_change"

        return None, None

    def add_completion_labels(self, df):
        """
            Add a 'completion_date' column for each file based on two strategies:
            1. A stable pattern: percentage_change stays below threshold for consecutive_days commits
            2. A deletion event:
            3. A long period of inactivity after the last commit (idle_days_cutoff)
        """
        df['completion_date'] = pd.NaT
        df['completion_reason'] = None

        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

        now = pd.Timestamp.now().normalize()

        project_cutoff = df["commit_interval_days"].replace(0, np.nan).dropna().quantile(0.95)
        project_cutoff = int(np.clip(project_cutoff, 30, 365))
        self.logging.info(f"Using project-wide inactivity cutoff of {project_cutoff} days")

        for path, group in df.groupby("path"):
            # Strategy 1
            completion_date, reason = self._check_stable_line_change_window(group)

            if completion_date:
                completion_date = pd.to_datetime(completion_date).tz_localize(None)
                df.loc[df["path"] == path, "completion_date"] = completion_date
                df.loc[df["path"] == path, "completion_reason"] = reason
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

            if days_since_last_commit > project_cutoff:
                df.loc[df["path"] == path, "completion_date"] = last_commit_date
                df.loc[df["path"] == path, "completion_reason"] = "idle_timeout"

        num_completed_files = df[df['completion_date'].notna()]['path'].nunique()
        total_files = df['path'].nunique()
        self.logging.info(
            f"Completed files: {num_completed_files} / {total_files} ({(num_completed_files / total_files * 100):.2f}%)")

        strategy_counts = (
            df[df['completion_reason'].notna()]
            .groupby("path")
            .first()["completion_reason"]
            .value_counts()
        )
        for reason, count in strategy_counts.items():
            self.logging.info(f"{reason}: {count} files")

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

    @staticmethod
    def select_snapshots(df, every="7D"):
        df = df.sort_values(["path", "date"])
        df["snapshot_bin"] = df["date"].dt.floor(every)
        latest_per_bin = (
            df.groupby(["path", "snapshot_bin"])
            .tail(1)
            .reset_index(drop=True)
            .drop(columns=["snapshot_bin"])
        )
        return latest_per_bin

    async def run(self, source_directory: str):
        """
        Fetch all files, compute features, and save them back to the database.
        """
        file_df = await self.fetch_all_files()
        file_df = file_df[file_df["path"].str.startswith(source_directory)].copy()
        file_features = self.calculate_metrics(file_df)
        file_features = self.select_snapshots(file_features, every="7D")

        feature_cols = [col for col in file_features.select_dtypes(include="number").columns
                        if col not in ["days_until_completion", "size", "cumulative_size"]]
        target_series = file_features["days_until_completion"]
        self.plotter.plot_feature_correlations(file_features[feature_cols], target_series)

        await self.save_features_to_db(file_features)

        return file_features