import logging

import pandas as pd
from kneed import KneeLocator
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from src.data_handling.async_database import AsyncDatabase


class ClusterAnalyser:
    def __init__(self, df_for_clustering, plotter, api_connection):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.df_for_clustering = df_for_clustering
        self.numerical_df_for_clustering = self.df_for_clustering[['cooccurrence_scaled', 'distance_scaled']]

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

        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(self.numerical_df_for_clustering)

        # Cluster labels are added to original combined_df
        self.df_for_clustering['cluster'] = cluster_labels

        # Save cluster labels to the database
        for file_path, cluster_id in zip(self.df_for_clustering['path'], cluster_labels):
            try:
                await AsyncDatabase.update_one(
                    self.api_connection.file_tracking_collection,
                    {'path': file_path},
                    {'$set': {'cluster': cluster_id}}
                )
            except Exception as e:
                logging.error(f"Failed to save cluster for {file_path}: {e}")

        logging.info("Cluster IDs saved to database.")

        self.analyse_clusters()

        logging.info("Plotting clusters")
        self.plotter.plot_clusters(self.df_for_clustering)

    def analyse_clusters(self):
        """
        Analyzes each cluster and saves visualisations and summaries for each cluster.
        """
        cluster_summary = {}
        for cluster in self.df_for_clustering['cluster'].unique():
            cluster_data = self.df_for_clustering[self.df_for_clustering['cluster'] == cluster]
            avg_cooccurrence = cluster_data['cooccurrence'].mean()
            avg_distance = cluster_data['distance'].mean()
            file_pairs = cluster_data[['file1', 'file2']].values

            cluster_summary[cluster] = {
                'avg_cooccurrence': avg_cooccurrence,
                'avg_distance': avg_distance,
                'file_count': len(file_pairs),
                'file_pairs': file_pairs.tolist()
            }

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

        summary_df = pd.DataFrame.from_dict(cluster_summary, orient='index')
        summary_df.to_csv('cluster_summary.csv')

        return summary_df

    def extract_features(self):
        features_list = []

        for file in self.df_for_clustering['file1'].unique():
            cluster = self.df_for_clustering.loc[self.df_for_clustering['file1'] == file, 'cluster'].values[0]
            co_occurrence = self.df_for_clustering.loc[self.df_for_clustering['file1'] == file, 'cooccurrence'].sum()

