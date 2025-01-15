import numpy as np
import pandas as pd

from src.data_handling.async_database import AsyncDatabase


class FileFeatureEngineer:

    def __init__(self, api_connection, project_name):
        self.api_connection = api_connection
        self.project_name = project_name

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
        file_df.dropna(subset=["size"], inplace=True)  # Drop rows where size is missing
        file_df.sort_values(["path", "date"], inplace=True)

        file_df.set_index("date", inplace=True)
        file_df.sort_index(inplace=True)

        grouped = file_df.groupby("path")
        metrics = []

        for path, group in grouped:
            group = group.copy()

            group["size"] = pd.to_numeric(group["size"], errors="coerce")
            group = group.dropna(subset=["size"])  # Drop rows where size is NaN

            features = {
                f"rolling_{window}_mean": group["size"].rolling(window=window).mean(),
                f"rolling_{window}_std": group["size"].rolling(window=window).std(),
                f"rolling_{window}_max": group["size"].rolling(window=window).max(),
                f"rolling_{window}_min": group["size"].rolling(window=window).min(),
                f"rolling_{window}_median": group["size"].rolling(window=window).median(),
                f"rolling_{window}_var": group["size"].rolling(window=window).var(),
                f"ema_{window}": group["size"].ewm(span=window, adjust=False).mean(),
            }

            for feature_name, feature_values in features.items():
                group[feature_name] = feature_values

            cumulative_features = {
                "cumulative_size": group["size"].cumsum(),
                "cumulative_mean": group["size"].expanding().mean(),
                "cumulative_std": group["size"].expanding().std(),
            }

            for feature_name, feature_values in cumulative_features.items():
                group[feature_name] = feature_values


            for lag in range(1, window + 1):
                group[f"lag_{lag}_size"] = group["size"].shift(lag)

            derived_features = {
                "absolute_change": group["size"].diff().abs(),
                "percentage_change": group["size"].pct_change() * 100,
                f"rolling_{window}_mean_to_std_ratio": group[f"rolling_{window}_mean"] / group[f"rolling_{window}_std"],
                }

            for feature_name, feature_values in derived_features.items():
                group[feature_name] = feature_values

            group["percentage_change"] = group["size"].pct_change().replace([np.inf, -np.inf], np.nan)
            group.replace([np.inf, -np.inf], 0, inplace=True)

            metrics.append(group)

        return pd.concat(metrics)

    async def save_features_to_db(self, file_features):
        """
        Save the computed features back to the database.
        """
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
        # Fetch all file data
        file_df = await self.fetch_all_files()

        # Calculate metrics
        file_features = self.calculate_metrics(file_df)

        # Save features back to the database
        await self.save_features_to_db(file_features)

        return file_features