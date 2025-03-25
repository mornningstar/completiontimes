import logging
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler

from src.data_handling.clustering.cluster_analyser import ClusterAnalyser
from src.visualisations.plotting import Plotter


def categorise(value, data, low_percentile=25, high_percentile=75):
    low_threshold = np.percentile(data, low_percentile)
    high_threshold = np.percentile(data, high_percentile)

    if value > high_threshold:
        return 'High'
    elif value > low_threshold:
        return 'Middle'
    else:
        return 'Low'


class FileCooccurenceAnalyser:
    def __init__(self, commit_data, project_name, api_connection, file_features=None):
        """
        Initialises the file co-occurrence analyser for specified commit data and project name.
        :param commit_data: Commit data from the database
        :param project_name: Name of the project
        """

        self.logging = logging.getLogger(self.__class__.__name__)
        self.scaler = None
        self.optimal_k = None
        self.commit_data = commit_data
        self.file_features = file_features
        self.api_connection = api_connection
        self.project_name = project_name

        self.plotter = Plotter(project_name=project_name)
        self.cooccurence_matrix = defaultdict(lambda: defaultdict(int))

    async def run(self, recluster=False):
        save_paths = {
            "combined_df": f"{self.project_name}/combined_df.parquet",
            "cooccurrence_df": f"{self.project_name}/cooccurrence_df.parquet",
            "cooccurrence_categorized_df": f"{self.project_name}/cooccurrence_categorized_df.parquet",
            "proximity_df": f"{self.project_name}/proximity_df.parquet",
        }

        output_dir = os.path.dirname(list(save_paths.values())[0])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            self.logging.info(f"Created directory: {output_dir}")

        if recluster or not any(os.path.exists(path) for path in save_paths.values()):
            self.logging.info("Generating new clustering...")

            cooccurrence_categorized_df, cooccurrence_df = self.build_cooccurrence_matrix()
            proximity_df = self.calculate_directory_proximity(cooccurrence_df)
            combined_df = self.combine_proximity_cooccurrence(proximity_df, cooccurrence_df)

            self.get_combined_data_matrix(combined_df)

            cluster_analyser = ClusterAnalyser(combined_df, self.plotter, self.api_connection)
            optimal_k = cluster_analyser.find_optimal_clusters()
            combined_df, summary_df = await cluster_analyser.run_clustering_analysis(k=optimal_k)

            csv_path = os.path.join(output_dir, 'cluster_summary.csv')
            summary_df.to_csv(csv_path, index=False)
            self.logging.info(f"Cluster summary saved to {csv_path}")

            self.logging.info("Saving final clustering data...")
            cooccurrence_df.to_parquet(save_paths["cooccurrence_df"])
            cooccurrence_categorized_df.to_parquet(save_paths["cooccurrence_categorized_df"])
            proximity_df.to_parquet(save_paths["proximity_df"])
            combined_df.to_parquet(save_paths["combined_df"])

        else:
            self.logging.info(f"Loading saved clustering dataframes from disk for project: {self.project_name}")
            cooccurrence_df = pd.read_parquet(save_paths["cooccurrence_df"])
            cooccurrence_categorized_df = pd.read_parquet(save_paths["cooccurrence_categorized_df"])
            proximity_df = pd.read_parquet(save_paths["proximity_df"])
            combined_df = pd.read_parquet(save_paths["combined_df"])

        return cooccurrence_df, cooccurrence_categorized_df, proximity_df, combined_df

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
        # Filter proximity_df and cooccurrence_df
        tracked_files = set(self.file_features['path'])

        # Filter proximity_df
        filtered_proximity_df = proximity_df[
            proximity_df['file1'].isin(tracked_files) & proximity_df['file2'].isin(tracked_files)
            ]
        # Deduplicate and remove self-pairs
        filtered_proximity_df = filtered_proximity_df[
            filtered_proximity_df['file1'] < filtered_proximity_df['file2']
            ]

        # Drop duplicated rows
        filtered_proximity_df = filtered_proximity_df.drop_duplicates(subset=['file1', 'file2'])

        # Filter cooccurrence_df
        filtered_cooccurrence_df = cooccurrence_df[
                                       cooccurrence_df.index.isin(tracked_files)
                                   ].loc[:, cooccurrence_df.columns.isin(tracked_files)]

        # Compute combined data
        filtered_proximity_df['cooccurrence'] = filtered_proximity_df.apply(
            lambda row: filtered_cooccurrence_df.at[row['file1'], row['file2']]
            if row['file1'] in filtered_cooccurrence_df.index and row['file2'] in filtered_cooccurrence_df.columns
            else 0,
            axis=1
        )

        cooccurrence_values = filtered_proximity_df['cooccurrence']
        distance_values = filtered_proximity_df['distance']

        filtered_proximity_df['distance_level'] = filtered_proximity_df['distance'].apply(
            lambda x: categorise(x, distance_values)
        )
        filtered_proximity_df['cooccurrence_level'] = filtered_proximity_df['cooccurrence'].apply(
            lambda x: categorise(x, cooccurrence_values)
        )

        # Merge file features
        combined_df = filtered_proximity_df.merge(
            self.file_features,
            left_on='file1',
            right_on='path',
            how='left'
        ).merge(
            self.file_features,
            left_on='file2',
            right_on='path',
            suffixes=('_file1', '_file2'),
            how='left'
        )

        combined_df["cumulative_size"] = (
                combined_df["cumulative_size_file1"] +
                combined_df["cumulative_size_file2"]
        )

        self.scaler = StandardScaler()
        combined_df[['cooccurrence_scaled', 'distance_scaled']] = self.scaler.fit_transform(
            combined_df[['cooccurrence', 'distance']]
        )

        nan_counts = combined_df.isnull().sum()
        self.logging.info(f"NaN counts after merge and scaling:\n{nan_counts[nan_counts > 0]}")

        return combined_df

    def plot(self, cooccurrence_df, cooccurrence_categorized_df, proximity_df, combined_df, plot_options=None):
        if plot_options is None:
            plot_options = {}
            logging.warning("No plot options provided. Skipping plots.")

        # Hierarchical Co-Occurrence
        if plot_options.get('hierarchical', False):
            self.plot_hierarchical_cooccurrence(cooccurrence_df)

        # Plot co-occurrence matrix
        if plot_options.get('cooccurrence_matrix', False):
            cooccurrence_data_type = plot_options.get('cooccurrence_data', 'categorised')
            top_n_files = plot_options.get('top_n_files', 15)

            if cooccurrence_data_type == 'categorised':
                logging.info("Using categorised co-occurrence data for the matrix plot.")
                category_to_num = {'Low': 0, 'Middle': 1, 'High': 2}
                cooccurrence_data = cooccurrence_categorized_df.apply(lambda col: col.map(category_to_num).fillna(0))
                value_label = 'Category'

            elif cooccurrence_data_type == 'raw':
                logging.info("Using raw co-occurrence data for the matrix plot.")
                cooccurrence_data = cooccurrence_df
                value_label = 'Co-occurrence'

            else:
                logging.error(
                    f"Invalid cooccurrence_data value: {cooccurrence_data_type}. Expected 'categorised' or 'raw'.")
                raise ValueError(f"Invalid cooccurrence_data: {cooccurrence_data_type}")

            # Call the plot function with the selected data
            self.plotter.plot_cooccurrence_matrix(
                cooccurrence_data,
                top_n_files=top_n_files,
                value_label=value_label
            )

        # Plot proximity matrix
        if plot_options.get('proximity_matrix', False):
            logging.info("Plotting proximity matrix.")
            self.plotter.plot_proximity_matrix(proximity_df)

        if plot_options.get('proximity_histogram', False):
            self.plotter.plot_proximity_histogram(proximity_df)

        # Plot distance vs. co-occurrence
        if plot_options.get('distance_vs_cooccurrence', False):
            distance_vs_cooccurrence_data = plot_options.get('distance_vs_cooccurrence_data', 'scaled')
            logging.info(f"Using {distance_vs_cooccurrence_data} data for distance vs. co-occurrence plot.")

            if distance_vs_cooccurrence_data == 'raw':
                self.plotter.plot_distance_vs_cooccurrence(combined_df, scaled=False)
            elif distance_vs_cooccurrence_data == 'scaled':
                self.plotter.plot_distance_vs_cooccurrence(combined_df, scaled=True)
            else:
                logging.error(f"Invalid distance_vs_cooccurrence_data: {distance_vs_cooccurrence_data}. "
                              f"Expected 'raw' or 'scaled'.")
                raise ValueError(f"Invalid distance_vs_cooccurrence_data: {distance_vs_cooccurrence_data}")

        # Plot Zipf distribution
        if plot_options.get('zipf_distribution', False):
            logging.info("Plotting Zipf distribution.")
            self.plotter.plot_zipf_distribution(cooccurrence_df)

    def plot_hierarchical_cooccurrence(self, cooccurrence_df):
        if cooccurrence_df.isnull().values.any():
            logging.warning("NaN values in cooccurrence_df. Filling with 0.")
            cooccurrence_df = cooccurrence_df.fillna(0)

        max_cooccurrence = cooccurrence_df.values.max()
        normalized_cooccurrence = cooccurrence_df.fillna(0) / max_cooccurrence

        distance_matrix = 1 - normalized_cooccurrence
        np.fill_diagonal(distance_matrix.values, 0)

        #distance_matrix = np.clip(distance_matrix, 0, 1)

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

        matrix_normalized = matrix / matrix.sum().sum()

        x_order = ['Low', 'Middle', 'High']
        y_order = ['High', 'Middle', 'Low']
        matrix_normalized = matrix_normalized.reindex(index=y_order, columns=x_order, fill_value=0)

        self.plotter.plot_distance_vs_cooccurrence_matrix(matrix_normalized)
