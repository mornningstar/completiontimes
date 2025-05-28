import logging

import numpy as np
import pandas as pd

from src.data_handling.database.file_repo import FileRepository
from src.data_handling.features.completion.completion_date_mixin import CompletionDateMixin
from src.visualisations.model_plotting import ModelPlotter


class BaseFeatureEngineer(CompletionDateMixin):
    def __init__(self, file_repo: FileRepository, plotter: ModelPlotter, use_categorical: bool = False):
        super().__init__()

        self.file_repo = file_repo
        self.plotter = plotter
        self.use_categorical = use_categorical

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

        if not self.use_categorical:
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

        df["std_dev_size_diff"] = df.groupby("path")["size_diff"].transform("std").fillna(0)

        roll = df.groupby("path")["size"].rolling(window=window)
        for stat in ["mean", "std", "max", "min", "median", "var"]:
            df[f"rolling_{window}_{stat}"] = getattr(roll.shift(1), stat)().reset_index(level=0, drop=True)

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

        if not self.use_categorical:
            committer_dummies = pd.get_dummies(df["committer_grouped"], prefix="committer")
            df = pd.concat([df, committer_dummies], axis=1)

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

    def calculate_metrics(self, df, window: int = 7):
        df = df.groupby("path").filter(lambda g: len(g) >= 5)

        df = self._add_metadata_features(df)
        df = self._add_time_series_stats(df, window=window)
        df = self._add_commit_activity_features(df, windows=[30, 90])
        df = self._add_temporal_dynamics_features(df)
        df = self._add_feature_interactions(df)
        df = self._add_committer_features(df)
        df = self._add_change_quality_features(df)
        df, num_completed_files, total_files = self.add_completion_labels(df)

        self.plotter.plot_completion_donut(num_completed_files, total_files)

        return df