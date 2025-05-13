import asyncio
import logging
import gc

import cupy as cp
import pandas as pd
from cuml.cluster import KMeans as cuKMeans
from kneed import KneeLocator
from sklearn.cluster import KMeans

from src.data_handling.database.async_database import AsyncDatabase
from src.gpu_lock import gpu_lock
from src.visualisations.predictions_plotting import PredictionsPlotter


class ClusterAnalyser:
    MAX_ROWS = 100_000

    def __init__(self, combined_df, plotter, api_connection):
        self.kmeans_optimal = None
        self.logging = logging.getLogger(self.__class__.__name__)
        self.combined_df = combined_df
        self.numerical_df_for_clustering = self.combined_df[['cooccurrence_scaled', 'distance_scaled']]

        self.plotter = plotter
        self.api_connection = api_connection

        self.optimal_k = None

    async def find_optimal_clusters(self, max_k=10):
        """
        Finds the optimal number of clusters for given data using the Elbow Method.
        :param max_k: Maximum number of clusters to try.
        """
        self.logging.info("Searching for optimal number of clusters...")

        inertia = []
        k_range = range(1, max_k + 1)
        models = {}

        if self.numerical_df_for_clustering.shape[0] > ClusterAnalyser.MAX_ROWS:
            self.logging.warning(
                f"Too many rows ({self.numerical_df_for_clustering.shape[0]}). Downsampling to {ClusterAnalyser.MAX_ROWS}.")
            df_for_clustering = self.numerical_df_for_clustering.sample(n=ClusterAnalyser.MAX_ROWS, random_state=42)
        else:
            df_for_clustering = self.numerical_df_for_clustering

        data = df_for_clustering.to_numpy().astype("float32")
        #data = self.numerical_df_for_clustering.to_numpy().astype('float32')

        if cp.cuda.is_available():
            async with gpu_lock:
                for k in k_range:
                    kmeans = cuKMeans(n_clusters=k, random_state=42)
                    kmeans.fit(data)
                    inertia.append(kmeans.inertia_)
                    models[k] = kmeans

            self.logging.debug("Freeing GPU memory after cuKMeans fitting.")
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
        else:
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(data)
                inertia.append(kmeans.inertia_)
                models[k] = kmeans

        # Using KneeLocator to find the "elbow" point
        knee_locator = KneeLocator(k_range, inertia, curve="convex", direction="decreasing")
        self.optimal_k = knee_locator.elbow

        if self.optimal_k is None:
            self.logging.warning("No clear elbow found, defaulting to max_k = %d", max_k)
            self.optimal_k = max_k

        self.kmeans_optimal = models[self.optimal_k]

        self.plotter.plot_elbow_method(k_range, inertia, self.optimal_k)

        self.logging.info("Found optimal number of clusters at k = {}".format(self.optimal_k))
        return self.optimal_k

    async def run_clustering_analysis(self):
        """
        Uses the pre-trained model from find_optimal_clusters to assign cluster labels.
        If k is provided, it is ignored; ensure find_optimal_clusters() was called before.
        """
        if self.kmeans_optimal is None:
            self.logging.error("No trained KMeans model available. Run find_optimal_clusters() first.")
            return

        self.logging.info("Running clustering analysis...")

        self.logging.info("Fitting KMeans on numerical data...")

        data = self.numerical_df_for_clustering.to_numpy().astype('float32')
        if cp.cuda.is_available():
            async with gpu_lock:
                cluster_labels = self.kmeans_optimal.predict(data)
        else:
            cluster_labels = self.kmeans_optimal.predict(data)

        self.combined_df['cluster'] = cluster_labels
        self.logging.info("Clustering completed.")

        self.logging.info("Updating clusters in the database for %d rows...", len(self.combined_df))

        unique_files = pd.concat([self.combined_df['file1'], self.combined_df['file2']]).unique()
        update_tasks = []

        for file in unique_files:
            rows_with_file = self.combined_df[
                (self.combined_df['file1'] == file) | (self.combined_df['file2'] == file)
                ]
            cluster_id = int(rows_with_file['cluster'].mode()[0])

            task = AsyncDatabase.update_one(
                self.api_connection.file_repo.file_tracking_collection,
                {'path': file},
                {'$set': {'cluster': cluster_id}}
            )

            update_tasks.append(task)

        batch_size = max(1, int(0.1 * len(update_tasks)))
        for i in range(0, len(update_tasks), batch_size):
            batch = update_tasks[i:i + batch_size]
            results = await asyncio.gather(*batch, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    self.logging.error("Error updating cluster for a file: %s", result)
                else:
                    self.logging.debug("Successfully updated cluster for a file.")

        self.logging.info("All cluster IDs updated in the database.")

        summary_df = self.analyse_clusters()

        predictions_plotter = PredictionsPlotter()
        predictions_plotter.plot_clusters(self.combined_df)

        self.logging.info("Clustering analysis completed.")
        return self.combined_df, summary_df

    def analyse_clusters(self):
        """
        Analyses each cluster and saves visualisations and summaries for each cluster.
        """
        if 'cooccurrence_scaled' not in self.combined_df.columns or 'distance_scaled' not in self.combined_df.columns:
            self.logging.error("Scaled columns are missing. Ensure data is scaled before analysis.")
            return

        cluster_agg = self.combined_df.groupby('cluster').agg(
            avg_cooccurrence=('cooccurrence', 'mean'),
            avg_distance=('distance', 'mean')
        ).reset_index()

        files_long = pd.concat([
            self.combined_df[['cluster', 'file1']].rename(columns={'file1': 'file'}),
            self.combined_df[['cluster', 'file2']].rename(columns={'file2': 'file'})
        ])

        file_counts = files_long.groupby('cluster')['file'].nunique().reset_index().rename(
            columns={'file': 'file_count'})

        summary_df = cluster_agg.merge(file_counts, on='cluster')

        self.plotter.plot_cluster_analysis(self.combined_df)

        return summary_df


    def extract_features(self):
        files_long = pd.concat([
            self.combined_df[['cluster', 'file1', 'cooccurrence']].rename(columns={'file1': 'file'}),
            self.combined_df[['cluster', 'file2', 'cooccurrence']].rename(columns={'file2': 'file'})
        ])

        features_df = files_long.groupby('file').agg(
            cluster=('cluster', lambda x: x.mode()[0] if not x.mode().empty else None),
            cooccurrence=('cooccurrence', 'sum')
        ).reset_index()

        return features_df