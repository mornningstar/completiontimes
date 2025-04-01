import logging

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from src.visualisations.plotting import Plotter


class ClusterPlotter(Plotter):
    """
    A class to handle all plotting needed by CluterAnalyser and FileCooccurrenceAnalyser.
    """

    def __init__(self, project_name):
        super().__init__(project_name)
        self.logging = logging.getLogger(self.__class__.__name__)
        self.project_name = project_name

    def plot_hierarchical_cooccurrence(self, cooccurrence_df):
        if cooccurrence_df.isnull().values.any():
            self.logging.warning("NaN values in cooccurrence_df. Filling with 0.")
            cooccurrence_df = cooccurrence_df.fillna(0)

        max_cooccurrence = cooccurrence_df.values.max()
        normalized_cooccurrence = cooccurrence_df.fillna(0) / max_cooccurrence

        distance_matrix = 1 - normalized_cooccurrence
        np.fill_diagonal(distance_matrix.values, 0)

        linked = linkage(squareform(distance_matrix), method='ward')

        plt.figure(figsize=(12, 8))
        dendrogram(linked, labels=cooccurrence_df.index, orientation='right', leaf_font_size=8)
        plt.title("File Clustering Dendrogram")
        plt.xlabel("Distance")
        plt.ylabel("Files")

        self.save_plot('hierarchical_cooccurrence.png')

    def plot_distance_vs_cooccurrence_matrix(self, matrix):
        plt.figure(figsize=(6, 6))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=False, linewidths=.5)
        plt.xlabel("Directory Distance Level")
        plt.ylabel("Co-occurrence Level")
        plt.title("Co-occurrence vs. Directory Distance Matrix")

        self.save_plot('matrix.png')

    def plot_zipf_distribution(self, cooccurrence_df):
        cooccurrence_dict = {}

        for i in cooccurrence_df.index:
            for j in cooccurrence_df.columns:
                if i != j:
                    # Ensure (i, j) and (j, i) are considered the same by ordering
                    pair = tuple(sorted((i, j)))
                    cooccurrence_value = cooccurrence_df.loc[i, j]

                    # Add to dictionary, summing values for duplicate pairs
                    if pair in cooccurrence_dict:
                        cooccurrence_dict[pair] += cooccurrence_value
                    else:
                        cooccurrence_dict[pair] = cooccurrence_value

        # Convert the dictionary to a DataFrame
        unique_pairs = list(cooccurrence_dict.keys())
        cooccurrence_values = list(cooccurrence_dict.values())

        cooccurrence_data = pd.DataFrame({'FilePair': unique_pairs, 'Cooccurrence': cooccurrence_values})

        # Format FilePair for plotting
        cooccurrence_data['FilePair'] = cooccurrence_data['FilePair'].apply(lambda x: f"{x[0]}, {x[1]}")

        # Filter for non-zero co-occurrences and sort by descending order
        cooccurrence_data = cooccurrence_data[cooccurrence_data['Cooccurrence'] > 0].sort_values(
            by='Cooccurrence', ascending=False)

        # Plot the data
        plt.figure(figsize=(12, 6))
        sns.barplot(data=cooccurrence_data.head(20), x='Cooccurrence', y='FilePair', hue='FilePair', dodge=False)
        plt.title('Zipf\'s Law for File Co-occurrence')
        plt.xlabel('Co-occurrence Count')
        plt.ylabel('File Pairs')
        plt.tight_layout()

        self.save_plot("zipf_distribution.png")

    def plot_distance_vs_cooccurrence(self, combined_df, scaled=True):
        plt.figure(figsize=(12, 8))

        distance_label = 'Directory Distance (Scaled)' if scaled else 'Directory Distance (Raw)'
        cooccurrence_label = 'Co-occurrence (Scaled)' if scaled else 'Co-occurrence (Raw)'
        filename = "distance_vs_cooccurrence_scaled.png" if scaled else "distance_vs_cooccurrence_raw.png"

        # Use 'hue' to color by cooccurrence_level and 'style' to differentiate by distance_level
        sns.scatterplot(
            data=combined_df,
            x='distance_scaled' if scaled else 'distance',
            y='cooccurrence_scaled' if scaled else 'cooccurrence',
            hue='cooccurrence_level',
            style='distance_level',
            palette='viridis',
            s=100  # increase point size for better visibility
        )

        plt.title(f'{distance_label} vs. {cooccurrence_label}')
        plt.xlabel(distance_label)
        plt.ylabel(cooccurrence_label)
        plt.legend(title='Levels')
        plt.tight_layout()

        self.save_plot(filename)

    def plot_proximity_histogram(self, proximity_df):
        plt.figure(figsize=(10, 6))
        sns.histplot(proximity_df['distance'], binwidth=1, kde=False)#, bins=30)
        plt.title('Distribution of File Directory Distances')
        plt.xlabel('Directory Distance')
        plt.ylabel('Frequency')
        plt.tight_layout()

        self.save_plot("proximity_histogram.png")

    def plot_proximity_matrix(self, proximity_df):
        proximity_pivot = proximity_df.pivot_table(index="file1", columns="file2", values="distance")

        plt.figure(figsize=(20, 16))
        sns.set_theme(font_scale=0.8)
        sns.heatmap(proximity_pivot, cmap='coolwarm', annot=False, square=True)
        plt.title('File Directory Proximity Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        self.save_plot("proximity_matrix.png")

    def plot_cooccurrence_matrix(self, cooccurrence_df, top_n_files=None, value_label="Value"):
        sorted_files = sorted(cooccurrence_df.index)
        numeric_matrix = cooccurrence_df.reindex(sorted_files, axis=0).reindex(sorted_files, axis=1)

        if top_n_files:
            cooccurrence_sums = numeric_matrix.sum(axis=1).sort_values(ascending=False)
            top_files = cooccurrence_sums.head(top_n_files).index
            numeric_matrix = numeric_matrix.loc[top_files, top_files]

            if numeric_matrix.empty:
                raise ValueError("Filtered matrix is empty.")

        plt.figure(figsize=(20, 16))
        sns.set_theme(font_scale=0.8)
        sns.heatmap(numeric_matrix, cmap='coolwarm', annot=False, square=True)
        plt.title('File Co-occurrence Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        plot_name = f"cooccurrence_matrix_{value_label.lower()}.png"
        self.save_plot(plot_name)

    def plot_cluster_analysis(self, combined_df):
        """
        Plots scatter plots for each cluster based on the scaled co-occurrence and distance values.
        """
        self.logging.info("Plotting scatter plots for all clusters...")

        for cluster, cluster_data in combined_df.groupby('cluster'):
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
            self.save_plot(f'cluster_{cluster}_analysis.png')

    def plot_elbow_method(self, k_range, inertia, optimal_k):
        plt.figure(figsize=(8, 6))
        plt.plot(k_range, inertia, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.xticks(k_range)
        plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
        plt.legend()

        self.save_plot("optimal_k.png")