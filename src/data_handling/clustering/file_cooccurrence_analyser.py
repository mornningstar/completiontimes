import gc
import logging
import os
from collections import defaultdict

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

from src.data_handling.clustering.cluster_analyser import ClusterAnalyser
from src.visualisations.cluster_plotting import ClusterPlotter


def categorise(value, low_threshold=25, high_threshold=75):
    if value > high_threshold:
        return 'High'
    elif value > low_threshold:
        return 'Middle'
    else:
        return 'Low'


class FileCooccurrenceAnalyser:
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

        self.plotter = ClusterPlotter(project_name=project_name)
        self.cooccurrence_matrix = defaultdict(lambda: defaultdict(int))

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
            await cluster_analyser.find_optimal_clusters()
            combined_df, summary_df = await cluster_analyser.run_clustering_analysis()

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
        self.logging.info("Building new co-occurrence matrix...")

        # 1. Flatten: Erstelle eine Liste von (commit, file)-Beziehungen
        rows = [
            {'commit': i, 'file': file['filename']}
            for i, commit in enumerate(self.commit_data)
            for file in commit.get('files', [])
        ]
        if not rows:
            self.logging.warning("No data in commit data found.")
            return pd.DataFrame(), pd.DataFrame()

        self.logging.debug("Finished 1.")

        # 2. Erstelle einen DataFrame aus den Beziehungen
        df = pd.DataFrame(rows)

        self.logging.debug("Finished 2.")

        # 3. Erstelle eine One-Hot Encoding Matrix: Zeilen = Commits, Spalten = Dateien
        # Dabei erhält jede Zelle den Wert 1, falls die Datei im Commit vorkommt.
        one_hot = pd.crosstab(df['commit'], df['file'])

        self.logging.debug("Finished 3.")

        # 4. Berechne die Co-Occurrence Matrix als X.T * X
        # Dabei zählt (i,j) die Anzahl der Commits, in denen beide Dateien vorkommen.
        if cp.cuda.is_available():
            self.logging.info("Using CUDA for building co-occurrence matrix.")

            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

            one_hot_cp = cp.asarray(one_hot.values, dtype=cp.float32)
            one_hot_sparse = cupy_csr_matrix(one_hot_cp)
            cooccurrence_sparse = one_hot_sparse.T.dot(one_hot_sparse)
            cooccurrence_matrix = cp.asnumpy(cooccurrence_sparse.toarray())

            self.logging.info("Freeing GPU memory after sparse matrix calculation.")
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
        else:
            self.logging.info("No CUDA available for building co-occurrence matrix.")

            one_hot_sparse = csr_matrix(one_hot.values.astype('float32'))
            cooccurrence_sparse = one_hot_sparse.T.dot(one_hot_sparse)
            cooccurrence_matrix = cooccurrence_sparse.toarray()

        #cooccurrence_matrix = one_hot.T.dot(one_hot)

        self.logging.debug("Finished 4.")

        # 5. Setze die Diagonale auf 0 (da wir nicht an Selbstpaaren interessiert sind)
        np.fill_diagonal(cooccurrence_matrix, 0)

        self.logging.debug("Finished 5.")

        # 6. Umwandeln in DataFrames mit passenden Indizes und Spalten
        cooccurrence_df = pd.DataFrame(cooccurrence_matrix, index=one_hot.columns, columns=one_hot.columns)

        self.logging.debug("Finished 6.")

        # 7. Kategorisierung: Wende die categorise-Funktion auf alle Werte an.
        #    Dabei wird das gesamte Wertefeld als Referenz für die Prozentile genutzt.
        all_values = cooccurrence_df.values.flatten()
        low_threshold, high_threshold = np.percentile(all_values, [25, 75])
        categorized = np.where(cooccurrence_df.values > high_threshold, 'High',
                               np.where(cooccurrence_df.values > low_threshold, 'Middle', 'Low'))
        cooccurrence_categorized_df = pd.DataFrame(categorized, index=cooccurrence_df.index,
                                                   columns=cooccurrence_df.columns)

        self.logging.debug("Finished 7.")

        return cooccurrence_categorized_df, cooccurrence_df

    def calculate_directory_proximity(self, cooccurrence_df):
        self.logging.info("Calculating directory proximity of co-occurrence matrix...")
        # Hole alle Dateipfade aus der Co-Occurrence-Matrix
        file_paths = cooccurrence_df.index.tolist()
        files = pd.Series(file_paths)
        depths = files.str.count(os.sep).values

        # 1. Berechne die Verzeichnistiefe (Anzahl der os.sep) für jeden Pfad
        depth_matrix = np.add.outer(depths, depths)
        split_paths = files.str.split(os.sep)

        # 2. Teile die Pfade in ihre Komponenten auf
        max_len = split_paths.map(len).max()
        split_df = pd.DataFrame(split_paths.tolist(), index=files, columns=range(max_len))

        # 3. Berechne die gemeinsame Verzeichnistiefe für jedes Dateipaar
        n = len(files)
        common_depth_matrix = np.zeros((n, n), dtype=int)
        mask = np.ones((n, n), dtype=bool)

        for level in range(max_len):
            level_vals = split_df[level].values
            eq_matrix = (level_vals[:, None] == level_vals[None, :])
            new_matches = mask & eq_matrix
            common_depth_matrix[new_matches] += 1
            mask = mask & eq_matrix

        # 4. Berechne die finale Distanzmatrix
        distance_matrix = depth_matrix - 2 * common_depth_matrix
        # Optional: Umwandeln in ein DataFrame, falls gewünscht
        distance_df = pd.DataFrame(distance_matrix, index=files, columns=files)

        return distance_df

    def combine_proximity_cooccurrence(self, distance_df, cooccurrence_df):
        self.logging.info("Combining proximity with co-occurrence matrices...")
        # 1. Filtere auf die interessanten Dateien
        tracked_files = set(self.file_features['path'])

        # Filtere die distance_df (unsere neue Matrix) auf die tracked_files
        filtered_distance_df = distance_df.loc[
            distance_df.index.intersection(tracked_files),
            distance_df.columns.intersection(tracked_files)
        ]

        # 2. Konvertiere die symmetrische Matrix in ein "langes" Format
        proximity_long = (
            filtered_distance_df.reset_index()
            .melt(id_vars='index', var_name='file2', value_name='distance')
            .rename(columns={'index': 'file1'})
        )

        # Entferne Selbstpaare und doppelte Paare (nur ein Eintrag pro Paar)
        proximity_long = proximity_long[proximity_long['file1'] < proximity_long['file2']]

        # 3. Filtere die cooccurrence_df ebenfalls auf die tracked_files
        filtered_cooccurrence_df = cooccurrence_df.loc[
            cooccurrence_df.index.intersection(tracked_files),
            cooccurrence_df.columns.intersection(tracked_files)
        ]

        df_reset = filtered_cooccurrence_df.reset_index()
        df_reset = df_reset.rename(columns={df_reset.columns[0]: 'file1'})

        cooccurrence_long = df_reset.melt(id_vars='file1', var_name='file2', value_name='cooccurrence')

        # Merge den langen Co‑Occurrence-DataFrame mit proximity_long
        proximity_long = proximity_long.merge(
            cooccurrence_long,
            on=['file1', 'file2'],
            how='left'
        )

        # 4. Kategorisiere die Werte (Nutzung der bestehenden categorise-Funktion)
        low_distance, high_distance = np.percentile(proximity_long['distance'], [25, 75])
        proximity_long['distance_level'] = proximity_long['distance'].apply(
            lambda x: categorise(x, low_distance, high_distance))

        low_cooccurrence, high_cooccurrence = np.percentile(proximity_long['cooccurrence'], [25, 75])
        proximity_long['cooccurrence_level'] = proximity_long['cooccurrence'].apply(
            lambda x: categorise(x, low_cooccurrence, high_cooccurrence))

        # 5. Füge File-Features zu den jeweiligen Dateien hinzu
        combined_df = proximity_long.merge(
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

        # 6. Skaliere die numerischen Spalten
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
            self.logging.warning("No plot options provided. Skipping plots.")

        # Hierarchical Co-Occurrence
        if plot_options.get('hierarchical', False):
            self.plotter.plot_hierarchical_cooccurrence(cooccurrence_df)

        # Plot co-occurrence matrix
        if plot_options.get('cooccurrence_matrix', False):
            cooccurrence_data_type = plot_options.get('cooccurrence_data', 'categorised')
            top_n_files = plot_options.get('top_n_files', 15)

            if cooccurrence_data_type == 'categorised':
                self.logging.info("Using categorised co-occurrence data for the matrix plot.")
                category_to_num = {'Low': 0, 'Middle': 1, 'High': 2}
                cooccurrence_data = cooccurrence_categorized_df.apply(lambda col: col.map(category_to_num).fillna(0))
                value_label = 'Category'

            elif cooccurrence_data_type == 'raw':
                self.logging.info("Using raw co-occurrence data for the matrix plot.")
                cooccurrence_data = cooccurrence_df
                value_label = 'Co-occurrence'

            else:
                self.logging.error(
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
            self.logging.info("Plotting proximity matrix.")
            self.plotter.plot_proximity_matrix(proximity_df)

        if plot_options.get('proximity_histogram', False):
            self.plotter.plot_proximity_histogram(proximity_df)

        # Plot distance vs. co-occurrence
        if plot_options.get('distance_vs_cooccurrence', False):
            distance_vs_cooccurrence_data = plot_options.get('distance_vs_cooccurrence_data', 'scaled')
            self.logging.info(f"Using {distance_vs_cooccurrence_data} data for distance vs. co-occurrence plot.")

            if distance_vs_cooccurrence_data == 'raw':
                self.plotter.plot_distance_vs_cooccurrence(combined_df, scaled=False)
            elif distance_vs_cooccurrence_data == 'scaled':
                self.plotter.plot_distance_vs_cooccurrence(combined_df, scaled=True)
            else:
                self.logging.error(f"Invalid distance_vs_cooccurrence_data: {distance_vs_cooccurrence_data}. "
                              f"Expected 'raw' or 'scaled'.")
                raise ValueError(f"Invalid distance_vs_cooccurrence_data: {distance_vs_cooccurrence_data}")

        # Plot Zipf distribution
        if plot_options.get('zipf_distribution', False):
            self.logging.info("Plotting Zipf distribution.")
            self.plotter.plot_zipf_distribution(cooccurrence_df)

    def get_combined_data_matrix(self, combined_df):
        matrix = (combined_df[['cooccurrence_level', 'distance_level']]
                  .groupby(['cooccurrence_level', 'distance_level']).size().unstack(fill_value=0))

        matrix_normalized = matrix / matrix.sum().sum()

        x_order = ['Low', 'Middle', 'High']
        y_order = ['High', 'Middle', 'Low']
        matrix_normalized = matrix_normalized.reindex(index=y_order, columns=x_order, fill_value=0)

        self.plotter.plot_distance_vs_cooccurrence_matrix(matrix_normalized)
