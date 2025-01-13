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
        file_df.ffill(inplace=True)  # Forward-fill missing values
        file_df.bfill(inplace=True)  # Backward-fill missing values
        file_df.dropna(subset=["size"], inplace=True)  # Drop rows where size is still NaN

        grouped = file_df.groupby("path")

        metrics = []
        for path, group in grouped:
            group = group.set_index("date")  # Use date as index for rolling calculations
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

            # Lag features
            for lag in range(1, window + 1):
                group[f"lag_{lag}_size"] = group["size"].shift(lag)

                # Derived features
            derived_features = {
                "absolute_change": group["size"].diff().abs(),
                "percentage_change": group["size"].pct_change() * 100,
                "rolling_mean_to_std_ratio": group[f"rolling_{window}_mean"] / group[f"rolling_{window}_std"],
                }

            for feature_name, feature_values in derived_features.items():
                group[feature_name] = feature_values

            group["rolling_mean_to_std_ratio"] = group["rolling_mean_to_std_ratio"].replace([np.inf, -np.inf], 0)

            metrics.append(group)

        # Combine all the groups back into a single DataFrame
        return pd.concat(metrics)

    async def save_features_to_db(self, file_features):
        """
        Save the computed features back to the database.
        """
        grouped_features = file_features.groupby("path")

        for path, group in grouped_features:
            features = group.to_dict(orient="records")  # Each record corresponds to a commit

            # Update the corresponding file entry in the database
            query = {"path": path}
            update = {"$set": {"features": features}}  # Save features as a list under "features"
            await AsyncDatabase.update_one(
                self.api_connection.file_tracking_collection, query, update
            )

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