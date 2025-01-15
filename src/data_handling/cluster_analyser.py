import logging

import pandas as pd
from kneed import KneeLocator
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from src.data_handling.async_database import AsyncDatabase


class ClusterAnalyser:
    def __init__(self, combined_df, plotter, api_connection):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.combined_df = combined_df
        self.numerical_df_for_clustering = self.combined_df[['cooccurrence_scaled', 'distance_scaled']]

        self.plotter = plotter
        self.api_connection = api_connection

        self.optimal_k = None

    def find_optimal_clusters(self, max_k=10):
        """
        Finds the optimal number of clusters for given data using the Elbow Method.
        :param max_k: Maximum number of clusters to try.
        """
        logging.info("Finding optimal number of clusters")
        inertia = []
        k_range = range(1, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.numerical_df_for_clustering)
            inertia.append(kmeans.inertia_)

        # Using KneeLocator to find the "elbow" point
        knee_locator = KneeLocator(k_range, inertia, curve="convex", direction="decreasing")
        self.optimal_k = knee_locator.elbow

        # Plot the Elbow Method (optional for visualization)
        plt.figure(figsize=(8, 6))
        plt.plot(k_range, inertia, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.xticks(k_range)
        plt.axvline(self.optimal_k, color='red', linestyle='--', label=f'Optimal k = {self.optimal_k}')
        plt.legend()

        self.plotter.save_plot("optimal_k.png")

        logging.info("Found optimal number of clusters at k = {}".format(self.optimal_k))
        return self.optimal_k

    async def run_clustering_analysis(self, k=4):
        """
        Performs KMeans clustering with the specified number of clusters (k) and
        analyses the resulting clusters.
        """
        logging.info("Running clustering analysis")

        logging.info("Initializing KMeans...")
        kmeans = KMeans(n_clusters=k, random_state=42)
        logging.info("Fitting KMeans on numerical data...")
        cluster_labels = kmeans.fit_predict(self.numerical_df_for_clustering)
        logging.info("Clustering completed.")

        self.combined_df['cluster'] = cluster_labels

        updated_files = set()
        logging.info("Updating clusters in the database for %d rows...", len(self.combined_df))

        unique_files = pd.concat([self.combined_df['file1'], self.combined_df['file2']]).unique()
        for file in unique_files:
            rows_with_file = self.combined_df[
                (self.combined_df['file1'] == file) | (self.combined_df['file2'] == file)
                ]
            cluster_id = int(rows_with_file['cluster'].mode()[0])
            try:
                await AsyncDatabase.update_one(
                    self.api_connection.file_tracking_collection,
                    {'path': file},
                    {'$set': {'cluster': cluster_id}}
                )
                logging.debug("Updated cluster ID %d for file %s", cluster_id, file)
            except Exception as e:
                logging.error(f"Failed to save cluster for {file}: {e}")

        logging.info("All cluster IDs updated in the database.")

        summary_df = self.analyse_clusters()

        logging.info("Plotting clusters...")
        self.plotter.plot_clusters(self.combined_df)

        logging.info("Clustering analysis completed.")
        return self.combined_df, summary_df

    def analyse_clusters(self):
        """
        Analyses each cluster and saves visualisations and summaries for each cluster.
        """
        cluster_summaries = []
        if 'cooccurrence_scaled' not in self.combined_df.columns or 'distance_scaled' not in self.combined_df.columns:
            self.logger.error("Scaled columns are missing. Ensure data is scaled before analysis.")
            return

        for cluster in self.combined_df['cluster'].unique():
            cluster_data = self.combined_df[self.combined_df['cluster'] == cluster]

            avg_cooccurrence = cluster_data['cooccurrence'].mean()
            avg_distance = cluster_data['distance'].mean()

            unique_files = pd.concat([cluster_data['file1'], cluster_data['file2']]).unique()
            file_count = len(unique_files)

            cluster_summaries.append({
                'cluster': cluster,
                'avg_cooccurrence': avg_cooccurrence,
                'avg_distance': avg_distance,
                'file_count': file_count
            })

            plt.figure(figsize=(10, 8))
            plt.scatter(
                cluster_data['cooccurrence_scaled'],
                cluster_data['distance_scaled'],
                alpha=0.6,
                label=f'Cluster {cluster}'
            )
            plt.xlabel('Co-occurrence (scaled)')
            plt.ylabel('Distance (scaled)')
            plt.title(f'Cluster {cluster} Analysis')
            plt.legend()
            self.plotter.save_plot(f'cluster_{cluster}_analysis.png')
            plt.close()

        summary_df = pd.DataFrame(cluster_summaries)
        #summary_df.to_csv('cluster_summary}.csv', index=False)

        return summary_df

    def extract_features(self):
        features_list = []

        unique_files = pd.concat([self.combined_df['file1'], self.combined_df['file2']]).unique()
        for file in unique_files:
            rows_with_file = self.combined_df[
                (self.combined_df['file1'] == file) | (self.combined_df['file2'] == file)
                ]
            cluster = rows_with_file['cluster'].mode()[0]  # Assign the most common cluster
            co_occurrence = rows_with_file['cooccurrence'].sum()

            features_list.append({'file': file, 'cluster': cluster, 'cooccurrence': co_occurrence})

        return pd.DataFrame(features_list)

