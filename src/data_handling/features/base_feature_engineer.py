import logging

import pandas as pd

from src.data_handling.database.file_repo import FileRepository
from src.data_handling.features.completion_date_labler import CompletionDateLabler
from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.visualisations.model_plotting import ModelPlotter
from src.data_handling.features.generators.commit_history_feature_generator import CommitHistoryFeatureGenerator
from src.data_handling.features.generators.file_path_feature_generator import FilePathFeatureGenerator
from src.data_handling.features.generators.file_size_feature_generator import FileSizeFeatureGenerator
from src.data_handling.features.generators.line_change_feature_generator import LineChangeFeatureGenerator
from src.data_handling.features.generators.committer_feature_generator import CommitterFeatureGenerator


# ALL_FEATURE_GENERATORS: dict[str, AbstractFeatureGenerator] = {
#     'MetaDataFeatures': FileMetadataFeatureGenerator(),
#     'TimeSeriesFeatures': TimeSeriesGenerator(),
#     'CommitActivityFeatures': CommitActivityFeatureGenerator(),
#     'TemporalDynamicsFeatures': TemporalDynamicsFeatureGenerator(),
#     'FeatureInteractions': FeatureInteractionsGenerator(),
#     'CommitterFeatures': CommitterFeatureGenerator(),
# }

class BaseFeatureEngineer:

    def __init__(self, file_repo: FileRepository, plotter: ModelPlotter, use_categorical: bool = False):
        super().__init__()

        self.file_repo = file_repo
        self.plotter = plotter
        self.use_categorical = use_categorical
        self.logging = logging.getLogger(self.__class__.__name__)
        self.completion_labler = CompletionDateLabler()

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

    def engineer_features(self, df, window: int = 7, include_sets = None):
        df = df.groupby("path").filter(lambda g: len(g) >= 5)
        if df.empty:
            return df

        if include_sets is None:
            include_sets = feature_generator_registry.get_all_names()

        for group_name in include_sets:
            generator = feature_generator_registry.get(group_name)
            if generator:
                self.logging.info(f"Generating features for: {group_name}")
                df = generator.generate(
                    df,
                    use_categorical=self.use_categorical,
                    windows=[30, 90]
                )
            else:
                self.logging.warning(f"Feature group '{group_name}' not found in registry.")

        # Apply the completion date labels
        df, num_completed_files, total_files = self.completion_labler.label(df)
        if total_files > 0:
            self.plotter.plot_completion_donut(num_completed_files, total_files)

        return df
