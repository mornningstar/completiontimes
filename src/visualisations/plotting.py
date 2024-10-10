import os
from itertools import cycle

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, project_name=None, images_dir='images'):
        self.images_dir = images_dir
        self.project_name = project_name

        self._create_directory(self.images_dir)

        self.project_images_dir = f"{self.images_dir}/{self.project_name}"
        self._create_directory(self.project_images_dir)

    @staticmethod
    def _create_directory(directory_path):
        """Helper function to create a directory if it does not exist."""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def save_plot(self, filename):
        """Helper function to save the current plot to the project directory."""
        plt.savefig(os.path.join(self.project_images_dir, filename))
        plt.close()

    def plot(self, df, title, ylabel):
        """
        Helper function to plot the given dataframe.
        :param df: dataframe to plot
        :param title: title of the plot
        :param ylabel: Label for the y-axis
        :return:
        """
        plt.figure(figsize=(12, 6))

        plt.plot(df.index, df.iloc[:, 0], marker='o', linestyle='-')
        plt.title(f'{title} for {self.project_name}')
        plt.xlabel('Time')
        plt.ylabel(ylabel)
        plt.grid(True)

        self.save_plot("plot.png")

    def plot_cooccurrence_matrix(self, cooccurrence_df, top_n_files=None):
        """
        Plots a heatmap of the co-occurrence matrix.
        :param cooccurrence_df:
        :param top_n_files: Number of top co-occurrence files to plot
        :return:
        """

        category_to_num = {'Low': 0, 'Middle': 1, 'High': 2}
        numeric_matrix = cooccurrence_df.apply(lambda col: col.map(category_to_num).fillna(0))

        sorted_files = sorted(cooccurrence_df.index)
        numeric_matrix = numeric_matrix.reindex(sorted_files, axis=0).reindex(sorted_files, axis=1)

        if top_n_files:
            cooccurrence_sums = numeric_matrix.sum(axis=1).sort_values(ascending=False)
            top_files = cooccurrence_sums.head(top_n_files).index
            numeric_matrix = numeric_matrix.loc[top_files, top_files]

            if numeric_matrix.empty:
                raise ValueError("Filtered matrix is empty.")

        plt.figure(figsize=(20, 16))
        sns.set(font_scale=0.8)
        sns.heatmap(numeric_matrix, cmap='coolwarm', annot=False, square=True)
        plt.title('File Co-occurrence Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        self.save_plot("cooccurrence_matrix.png")

    def plot_zipf_distribution(self, cooccurrence_df):
        file_pairs = [(i, j) for i in cooccurrence_df.index for j in cooccurrence_df.columns]
        cooccurrence_values = cooccurrence_df.values.flatten()
        cooccurrence_data = pd.DataFrame({'FilePair': file_pairs, 'Cooccurrence': cooccurrence_values})

        cooccurrence_data['FilePair'] = cooccurrence_data['FilePair'].apply(lambda x: f"{x[0]}, {x[1]}")

        cooccurrence_data = cooccurrence_data[cooccurrence_data['Cooccurrence'] > 0].sort_values(
            by='Cooccurrence', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=cooccurrence_data.head(20), x='Cooccurrence', y='FilePair', hue='FilePair', dodge=False)
        plt.title('Zipf\'s Law for File Co-occurrence')
        plt.xlabel('Co-occurrence Count')
        plt.ylabel('File Pairs')
        plt.tight_layout()

        self.save_plot("zipf_distribution.png")

    def plot_proximity_matrix(self, proximity_df):
        proximity_pivot = proximity_df.pivot_table(index="file1", columns="file2", values="distance")

        plt.figure(figsize=(20, 16))
        sns.set(font_scale=0.8)
        sns.heatmap(proximity_pivot, cmap='coolwarm', annot=False, square=True)
        plt.title('File Directory Proximity Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        self.save_plot("proximity_matrix.png")

    def plot_proximity_histogram(self, proximity_df):
        plt.figure(figsize=(10, 6))
        sns.histplot(proximity_df['distance'], bins=30, kde=False)
        plt.title('Distribution of File Directory Distances')
        plt.xlabel('Directory Distance')
        plt.ylabel('Frequency')
        plt.tight_layout()

        self.save_plot("proximity_histogram.png")

    def plot_distance_vs_cooccurrence(self, combined_df):
        plt.figure(figsize=(12, 8))

        # Use 'hue' to color by cooccurrence_level and 'style' to differentiate by distance_level
        sns.scatterplot(
            data=combined_df,
            x='distance',
            y='cooccurrence',
            hue='cooccurrence_level',
            style='distance_level',
            palette='viridis',
            s=100  # increase point size for better visibility
        )

        plt.title('Directory Distance vs. Co-occurrence')
        plt.xlabel('Directory Distance')
        plt.ylabel('Co-occurrence')
        plt.legend(title='Levels')
        plt.tight_layout()

        self.save_plot("distance_vs_cooccurrence.png")

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

        self.save_plot("commits.png")

    def plot_commit_predictions(self, commits_df, model_info, task):

        plt.figure(figsize=(12, 6))

        plt.plot(commits_df.index, commits_df[task], label=f'Historical {task.capitalize()}', linestyle='-', marker='o',
                 color='blue')

        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_cycle = cycle(colors)

        for model_name, info in model_info.items():
            x_train_dates = pd.to_datetime(info['x_train'].flatten()).tz_localize(None)
            x_test_dates = pd.to_datetime(info['x_test'].flatten()).tz_localize(None)

            predictions = info['predictions'].values if isinstance(info['predictions'], pd.Series) else info[
                'predictions']
            predicted_df = pd.DataFrame({task: predictions}, index=x_test_dates)

            current_colour = next(color_cycle)

            plt.plot(predicted_df.index, predicted_df[task],
                     label=f'Prediction by {model_name} (MSE: {info["mse"]:.2f}, MAE: {info["mae"]:.2f}, '
                           f'RMSE: {info["rmse"]:.2f})', linestyle='--', marker='o', color=current_colour)

            last_actual_date = x_train_dates[-1]
            last_actual_value = commits_df[task].loc[last_actual_date]
            first_pred_value = predicted_df.iloc[0][task]
            plt.plot([last_actual_date, x_test_dates[0]], [last_actual_value, first_pred_value], color=current_colour,
                     linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'{task.capitalize()} Over Time with Forecast')
        plt.grid(True)
        plt.legend()

        self.save_plot(f'commit_predictions_{task}.png')

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

        self.save_plot(f'predictions_{file_path.replace("/", "_")}.png')
