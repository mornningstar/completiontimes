import os
from collections import defaultdict
from itertools import combinations

import pandas as pd

from src.visualisations.plotting import Plotter


class FileCooccurenceAnalyser:
    def __init__(self, commit_data, project_name):
        self.commit_data = commit_data

        self.plotter = Plotter(project_name=project_name)

        self.cooccurence_matrix = defaultdict(lambda: defaultdict(int))
        self.cooccurence_df = None
        self.proximity_df = None
        self.combined_df = None

    def run(self):
        self.build_cooccurrence_matrix()
        self.calculate_directory_proximity()
        self.combine_proximity_cooccurrence()

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

        self.cooccurence_df = pd.DataFrame(self.cooccurence_matrix).fillna(0)
        self.cooccurence_df = (self.cooccurence_df + self.cooccurence_df.T) / 2  # Ensure symmetry

    def calculate_directory_proximity(self):
        def directory_depth(file_path):
            return file_path.count(os.sep)

        def common_directory_depth(file1, file2):
            common_path = os.path.commonpath([file1, file2])
            return common_path.count(os.sep)

        proximity_data = []

        for file1 in self.cooccurence_df.index:
            for file2 in self.cooccurence_df.columns:
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

        self.proximity_df = pd.DataFrame(proximity_data)

    def combine_proximity_cooccurrence(self):
        combined_data = []

        for _, row in self.proximity_df.iterrows():
            file1 = row['file1']
            file2 = row['file2']
            distance = row['distance']
            cooccurrence = self.cooccurence_df.at[
                file1, file2] if file1 in self.cooccurence_df.index and file2 in self.cooccurence_df.columns else 0

            combined_data.append({
                'file1': file1,
                'file2': file2,
                'distance': distance,
                'cooccurrence': cooccurrence
            })

        self.combined_df = pd.DataFrame(combined_data)

    def plot(self):
        self.plotter.plot_cooccurrence_matrix(self.cooccurence_df)
        self.plotter.plot_proximity_matrix(self.proximity_df)
        self.plotter.plot_proximity_histogram(self.proximity_df)
        self.plotter.plot_distance_vs_cooccurrence(self.combined_df)
