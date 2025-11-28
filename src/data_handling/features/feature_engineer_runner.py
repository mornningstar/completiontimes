from src.data_handling.data_loader import DataLoader


class FeatureEngineerRunner:
    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer
        self.data_loader = DataLoader(feature_engineer.file_repo)

    async def run(self, source_directory: str, is_static: bool = False, include_sets = None):
        """
        Fetch all files, compute features, and save them back to the database.
        """
        file_df = await self.data_loader.fetch_all_files()
        file_df = file_df[file_df["path"].str.startswith(source_directory)].copy()

        file_features, categorical_cols = self.feature_engineer.engineer_features(file_df, include_sets=include_sets)
        target_series = file_features["days_until_completion"]
        feature_cols = [col for col in file_features.select_dtypes(include="number").columns
                        if col not in ["days_until_completion", "size", "cumulative_size"]]
        self.feature_engineer.plotter.plot_feature_correlations(file_features[feature_cols], target_series)

        await self.feature_engineer.save_features_to_db(file_features)

        return file_features, categorical_cols
