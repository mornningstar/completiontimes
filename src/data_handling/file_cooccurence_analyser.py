import os
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler

from src.data_handling.cluster_analyser import ClusterAnalyser
from src.visualisations.plotting import Plotter


def categorise(value, max_value):
    if value > 0.5 * max_value:
        return 'High'
    elif value > 0.25 * max_value:
        return 'Middle'
    else:
        return 'Low'


class FileCooccurenceAnalyser:
    def __init__(self, commit_data, project_name):
        """
        Initialises the file co-occurrence analyser for specified commit data and project name.
        :param commit_data: Commit data from the database
        :param project_name: Name of the project
        """

        self.optimal_k = None
        self.commit_data = commit_data

        self.plotter = Plotter(project_name=project_name)

        self.cooccurence_matrix = defaultdict(lambda: defaultdict(int))

    def run(self):
        print("Running file co-occurrence analyser")

        cooccurrence_categorized_df, cooccurrence_df = self.build_cooccurrence_matrix()
        proximity_df = self.calculate_directory_proximity(cooccurrence_df)
        combined_df = self.combine_proximity_cooccurrence(proximity_df, cooccurrence_df)

        self.plot(cooccurrence_df, cooccurrence_categorized_df, proximity_df, combined_df)
        self.plot_hierarchical_cooccurrence(cooccurrence_df)

        self.get_combined_data_matrix(combined_df)

        cluster_analyser = ClusterAnalyser(combined_df, self.plotter)
        cluster_analyser.find_optimal_clusters()
        cluster_analyser.run_clustering_analysis()

    def build_cooccurrence_matrix(self):

        for commit in self.commit_data:
            try:
                files = [file['filename'] for file in commit['files']]
                for file1, file2 in combinations(files, 2):
                    self.cooccurence_matrix[file1][file2] += 1
                    self.cooccurence_matrix[file2][file1] += 1
            except KeyError:
                pass

        all_files = set(self.cooccurence_matrix.keys())
        for file in all_files:
            self.cooccurence_matrix[file][file] = 0  # Ensure diagonal is zero

        cooccurrence_df = pd.DataFrame(self.cooccurence_matrix).fillna(0)
        cooccurrence_df = (cooccurrence_df + cooccurrence_df.T) / 2  # Ensure symmetry

        # Prepare a categorized matrix for plotting
        max_cooccurrence = cooccurrence_df.values.max()

        cooccurrence_categorized_df = cooccurrence_df.apply(
            lambda col: col.map(lambda x: categorise(x, max_cooccurrence)))

        return cooccurrence_categorized_df, cooccurrence_df

    def calculate_directory_proximity(self, cooccurrence_df):
        def directory_depth(file_path):
            return file_path.count(os.sep)

        def common_directory_depth(file1, file2):
            common_path = os.path.commonpath([file1, file2])
            return common_path.count(os.sep)

        proximity_data = []

        for file1 in cooccurrence_df.index:
            for file2 in cooccurrence_df.columns:
                if file1 != file2:
                    file1_depth = directory_depth(file1)
                    file2_depth = directory_depth(file2)

                    common_depth = common_directory_depth(file1, file2)
                    distance = (file1_depth + file2_depth) - 2 * common_depth

                    proximity_data.append({
                        'file1': file1,
                        'file2': file2,
                        'distance': distance
                    })

        return pd.DataFrame(proximity_data)

    def combine_proximity_cooccurrence(self, proximity_df, cooccurrence_df):
        combined_data = []
        max_cooccurrence = cooccurrence_df.values.max()
        max_distance = proximity_df['distance'].max()

        for _, row in proximity_df.iterrows():
            file1 = row['file1']
            file2 = row['file2']
            distance = row['distance']
            cooccurrence = cooccurrence_df.at[
                file1, file2] if file1 in cooccurrence_df.index and file2 in cooccurrence_df.columns else 0

            combined_data.append({
                'file1': file1,
                'file2': file2,
                'distance': distance,
                'distance_level': categorise(distance, max_distance),
                'cooccurrence': cooccurrence,
                'cooccurrence_level': categorise(cooccurrence, max_cooccurrence)
            })

        combined_df = pd.DataFrame(combined_data)

        scaler = StandardScaler()
        combined_df[['cooccurrence_scaled', 'distance_scaled']] = scaler.fit_transform(
            combined_df[['cooccurrence', 'distance']]
        )

        return combined_df

    def plot(self, cooccurrence_df, cooccurrence_categorized_df, proximity_df, combined_df):
        self.plotter.plot_cooccurrence_matrix(cooccurrence_categorized_df, top_n_files=15)
        self.plotter.plot_proximity_matrix(proximity_df)
        self.plotter.plot_proximity_histogram(proximity_df)
        self.plotter.plot_distance_vs_cooccurrence(combined_df)
        self.plotter.plot_zipf_distribution(cooccurrence_df)

    def plot_hierarchical_cooccurrence(self, cooccurrence_df):
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

        self.plotter.save_plot('hierarchical_cooccurrence.png')

    def get_combined_data_matrix(self, combined_df):
        matrix = (combined_df[['cooccurrence_level', 'distance_level']]
                  .groupby(['cooccurrence_level', 'distance_level']).size().unstack(fill_value=0))

        x_order = ['Low', 'Middle', 'High']
        y_order = ['High', 'Middle', 'Low']
        matrix = matrix.reindex(index=y_order, columns=x_order, fill_value=0)

        self.plotter.plot_distance_vs_cooccurrence_matrix(matrix)
