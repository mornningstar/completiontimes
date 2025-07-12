import logging

import pandas as pd

from src.data_handling.database.file_repo import FileRepository
from src.data_handling.features.mixins.change_quality_features import ChangeQualityFeatureMixin
from src.data_handling.features.mixins.commit_activity_features import CommitActivityFeatureMixin
from src.data_handling.features.mixins.committer_features import CommitterFeatureMixin
from src.data_handling.features.mixins.completion_date_mixin import CompletionDateMixin
from src.data_handling.features.mixins.feature_interactions import FeatureInteractionsMixin
from src.data_handling.features.mixins.metadata_features import MetadataFeatureMixin
from src.data_handling.features.mixins.temporal_dynamics_features import TemporalDynamicsFeatureMixin
from src.data_handling.features.mixins.time_series_features import TimeSeriesFeatureMixin
from src.visualisations.model_plotting import ModelPlotter


ALL_FEATURE_GROUPS = [
    'MetaDataFeatures',
    'TimeSeriesFeatures',
    'CommitActivityFeatures',
    'TemporalDynamicsFeatures',
    'FeatureInteractions',
    'CommitterFeatures',
    'ChangeQualityFeatures'
]

class BaseFeatureEngineer(CompletionDateMixin, MetadataFeatureMixin, TimeSeriesFeatureMixin, CommitActivityFeatureMixin,
    TemporalDynamicsFeatureMixin, FeatureInteractionsMixin, CommitterFeatureMixin, ChangeQualityFeatureMixin,):

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

    async def save_features_to_db(self, file_features):
        """
        Save the computed features back to the database.
        """
        if "completion_date" in file_features.columns:
            file_features['completion_date'] = file_features['completion_date'].astype(object).where(
                file_features['completion_date'].notnull(), None
            )

        grouped_features = file_features.groupby("path")

        for path, group in grouped_features:
            features = group.reset_index().to_dict(orient="records")
            await self.file_repo.append_features_to_file(path, features, upsert=False)

    def collapse_to_first_last(self, df: pd.DataFrame, base_cols: list[str] | None = None) -> pd.DataFrame:
        known_static = ["file_extension", "path_depth", "in_test_dir", "in_docs_dir", "is_config_file", "is_markdown",
            "is_desktop_entry", "is_workflow_file", "has_readme_name", "is_source_code", "is_script"]
        static_cols = [col for col in known_static if col in df.columns]

        if base_cols is None:
            base_cols = ["size"]

        df_sorted = df.sort_values(["path", "date"])

        first_rows = df_sorted.groupby("path").first().reset_index()
        last_rows = df_sorted.groupby("path").last().reset_index()

        first_rows = first_rows[["path"] + base_cols].add_suffix("_first")
        first_rows = first_rows.rename(columns={"path_first": "path"})

        last_rows = last_rows[["path"] + base_cols].add_suffix("_last")

        snap_first_last = first_rows.merge(last_rows, on="path", how="inner")
        for col in base_cols:
            snap_first_last[f"{col}_diff_total"] = snap_first_last[f"{col}_last"] - snap_first_last[f"{col}_first"]

        static = df.groupby("path").first().reset_index()[["path"] + static_cols]

        final_dataset = static.merge(snap_first_last, on="path")

        return final_dataset

    def calculate_metrics(self, df, window: int = 7, include_sets = None):
        df = df.groupby("path").filter(lambda g: len(g) >= 5)

        if not include_sets:
            include_sets = ALL_FEATURE_GROUPS

        if 'MetaDataFeatures' in include_sets:
            df = self._add_metadata_features(df, use_categorical = self.use_categorical)
        if 'TimeSeriesFeatures' in include_sets:
            df = self._add_time_series_stats(df, window=window)
        if 'CommitActivityFeatures' in include_sets:
            df = self._add_commit_activity_features(df, windows=[30, 90])
        if 'TemporalDynamicsFeatures' in include_sets:
            df = self._add_temporal_dynamics_features(df)
        if 'FeatureInteractions' in include_sets:
            df = self._add_feature_interactions(df)
        if 'CommitterFeatures' in include_sets:
            df = self._add_committer_features(df, use_categorical = self.use_categorical)
        if 'ChangeQualityFeatures' in include_sets:
            df = self._add_change_quality_features(df)

        df, num_completed_files, total_files = self.add_completion_labels(df)

        self.plotter.plot_completion_donut(num_completed_files, total_files)

        return df