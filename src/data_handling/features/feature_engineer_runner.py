from src.data_handling.features.base_feature_engineer import BaseFeatureEngineer


class FeatureEngineerRunner:
    def __init__(self, feature_engineer: BaseFeatureEngineer):
        self.feature_engineer = feature_engineer

    async def run(self, source_directory: str):
        """
        Fetch all files, compute features, and save them back to the database.
        """
        file_df = await self.feature_engineer.fetch_all_files()
        file_df = file_df[file_df["path"].str.startswith(source_directory)].copy()
        file_features = self.feature_engineer.calculate_metrics(file_df)
        file_features = self.feature_engineer.select_snapshots(file_features, every="7D")

        feature_cols = [col for col in file_features.select_dtypes(include="number").columns
                        if col not in ["days_until_completion", "size", "cumulative_size"]]
        target_series = file_features["days_until_completion"]
        self.feature_engineer.plotter.plot_feature_correlations(file_features[feature_cols], target_series)

        await self.feature_engineer.save_features_to_db(file_features)

        return file_features
