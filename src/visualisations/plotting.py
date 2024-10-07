import os
from itertools import cycle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, project_name=None, images_dir='images'):
        self.images_dir = images_dir
        self.project_name = project_name

        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        self.project_images_dir = f"{self.images_dir}/{self.project_name}"
        if not os.path.exists(self.project_images_dir):
            os.makedirs(self.project_images_dir)

    def plot(self, df, title, ylabel):
        plt.figure(figsize=(12, 6))

        plt.plot(df.index, df.iloc[:, 0], marker='o', linestyle='-')
        plt.title(f'{title} for {self.project_name}')
        plt.xlabel('Time')
        plt.ylabel(ylabel)
        plt.grid(True)

        plt.savefig(f'{self.project_images_dir}/plot.png')
        plt.close()

    def plot_cooccurrence_matrix(self, cooccurrence_df, top_n_files=None):
        sorted_files = sorted(cooccurrence_df.index)
        cooccurrence_df = cooccurrence_df.reindex(sorted_files, axis=0)
        cooccurrence_df = cooccurrence_df.reindex(sorted_files, axis=1)

        if top_n_files:
            cooccurrence_sums = cooccurrence_df.sum(axis=1).sort_values(ascending=False)
            top_files = cooccurrence_sums.head(top_n_files).index
            cooccurrence_df = cooccurrence_df.loc[top_files, top_files]

        plt.figure(figsize=(20, 16))
        sns.set(font_scale=0.8)
        sns.heatmap(cooccurrence_df, cmap='coolwarm', annot=False, square=True)
        plt.title('File Co-occurrence Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(f'{self.project_images_dir}/plot_coocurrence_matrix.png')
        plt.close()

        self.plot_zipf_distribution(cooccurrence_df)

    def plot_zipf_distribution(self, cooccurrence_df):
        # Flatten the matrix to count file pairs and plot them
        file_pairs = [(i, j) for i in cooccurrence_df.index for j in cooccurrence_df.columns]
        cooccurrence_values = cooccurrence_df.values.flatten()
        cooccurrence_data = pd.DataFrame({'FilePair': file_pairs, 'Cooccurrence': cooccurrence_values})

        # Convert tuple to string for plotting
        cooccurrence_data['FilePair'] = cooccurrence_data['FilePair'].apply(lambda x: f"{x[0]}, {x[1]}")

        # Filter non-zero co-occurrence values and sort
        cooccurrence_data = cooccurrence_data[cooccurrence_data['Cooccurrence'] > 0].sort_values(
            by='Cooccurrence', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=cooccurrence_data.head(20), x='Cooccurrence', y='FilePair', hue='FilePair', dodge=False)
        plt.title('Zipf\'s Law for File Co-occurrence')
        plt.xlabel('Co-occurrence Count')
        plt.ylabel('File Pairs')
        plt.tight_layout()
        plt.savefig(f'{self.project_images_dir}/zipf_distribution.png')
        plt.close()

    def plot_proximity_matrix(self, proximity_df):
        proximity_pivot = proximity_df.pivot_table(index="file1", columns="file2", values="distance")

        plt.figure(figsize=(20, 16))
        sns.set(font_scale=0.8)
        sns.heatmap(proximity_pivot, cmap='coolwarm', annot=False, square=True)
        plt.title('File Directory Proximity Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(f'{self.project_images_dir}/plot_proximity_matrix.png')
        plt.close()

    def plot_proximity_histogram(self, proximity_df):
        plt.figure(figsize=(10, 6))
        sns.histplot(proximity_df['distance'], bins=30, kde=False)
        plt.title('Distribution of File Directory Distances')
        plt.xlabel('Directory Distance')
        plt.ylabel('Frequency')
        plt.tight_layout()

        plt.savefig(f'{self.project_images_dir}/plot_proximity_histogram.png')
        plt.close()

    def plot_distance_vs_cooccurrence(self, combined_df):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=combined_df, x='distance', y='cooccurrence')
        plt.title('Directory Distance vs. Co-occurrence')
        plt.xlabel('Directory Distance')
        plt.ylabel('Co-occurrence')
        plt.tight_layout()

        plt.savefig(f'{self.project_images_dir}/plot_distance_vs_cooccurrence.png')
        plt.close()

    def plot_commits(self, data, stats_to_plot):
        plt.figure(figsize=(12, 6))

        for stat in stats_to_plot:
            if stat in data.columns:
                plt.plot(data.index, data[stat], label=stat.capitalize())
            else:
                raise ValueError(f'Stat {stat} does not exist in the data.')

        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.title('Changes Over Time')
        plt.grid(True)
        plt.legend()

        plt.savefig(f'{self.project_images_dir}/plot_{self.project_name.replace("/", "_")}.png')
        plt.close()

    def plot_predictions(self, size_df, model_info, file_path):
        plt.figure(figsize=(12, 6))

        plt.plot(size_df.index, size_df['size'], label='Historical File Size', linestyle='-', marker='o', color='blue')

        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_cycle = cycle(colors)

        for model_name, info in model_info.items():
            x_train_dates = pd.to_datetime(info['x_train'].flatten())
            x_test_dates = pd.to_datetime(info['x_test'].flatten())

            last_train_point = info['y_train'].iloc[-1]
            last_train_date = x_train_dates[-1]

            predictions = info['predictions'].values if isinstance(info['predictions'], pd.Series) else info[
                'predictions']

            predicted_df = pd.DataFrame({'size': predictions}, index=x_test_dates)

            current_colour = next(color_cycle)

            plt.plot(predicted_df.index, predicted_df['size'],
                     label=f'Prediction by {model_name} (MSE: {info["mse"]:.2f})', linestyle='--', marker='o',
                     color=current_colour)

            if not predicted_df.empty:
                first_pred_size = predicted_df.iloc[0]['size']
                plt.plot([last_train_date, x_test_dates[0]], [last_train_point, first_pred_size], color=current_colour,
                         linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('File Size')
        plt.title(f'File Size Over Time for {file_path}')
        plt.grid(True)
        plt.legend()

        plt.savefig(f'{self.project_images_dir}/{file_path.replace("/", "_")}_plot.png')
        plt.close()
