import logging
import os
from itertools import cycle

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, project_name=None, images_dir='images'):
        self.logging = logging.getLogger(self.__class__.__name__)
        self.images_dir = images_dir
        self.project_name = project_name

        self._create_directory(self.images_dir)

        self.project_images_dir = f"{self.project_name}/{self.images_dir}"
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

    def plot_cooccurrence_matrix(self, cooccurrence_df, top_n_files=None, value_label="Value"):
        """
        Plots a heatmap of the co-occurrence matrix.
        :param cooccurrence_df:
        :param top_n_files: Number of top co-occurrence files to plot
        :return:
        """

        #category_to_num = {'Low': 0, 'Middle': 1, 'High': 2}
        #numeric_matrix = cooccurrence_df.apply(lambda col: col.map(category_to_num).fillna(0))

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
        logging.info(f"Co-occurrence matrix plot saved as {plot_name}.")

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
        sns.histplot(proximity_df['distance'], binwidth=1, kde=False)#, bins=30)
        plt.title('Distribution of File Directory Distances')
        plt.xlabel('Directory Distance')
        plt.ylabel('Frequency')
        plt.tight_layout()

        self.save_plot("proximity_histogram.png")

    def plot_distance_vs_cooccurrence(self, combined_df, scaled=True):
        """
        Uses the scaled distance and co-occurrence.
        :param combined_df:
        :return:
        """
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

    def plot_distance_vs_cooccurrence_matrix(self, matrix):
        plt.figure(figsize=(6, 6))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=False, linewidths=.5)
        plt.xlabel("Directory Distance Level")
        plt.ylabel("Co-occurrence Level")
        plt.title("Co-occurrence vs. Directory Distance Matrix")

        self.save_plot('matrix.png')

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

    def plot_lstm_predictions(self, commits_df, model_info, task):
        plt.figure(figsize=(12, 6))

        # Extract LSTMModel predictions from the model_info
        lstm_info = model_info['LSTMModel']
        predictions = lstm_info['predictions']
        x_train = lstm_info['x_train']
        y_train = lstm_info['y_train']
        y_test = lstm_info['y_test']

        # Ensure predictions and y_test are aligned
        assert len(predictions) == len(y_test), "Predictions and y_test must have the same length."

        # Plot the full actual data from commits_df
        plt.plot(commits_df.index, commits_df[task], label="Actual Values", color='blue')

        # Plot only the predictions for the test portion
        prediction_dates = commits_df.index[-len(y_test):]  # Get the dates corresponding to the test set
        plt.plot(prediction_dates, predictions, label="LSTMModel Predictions", color='red')

        # Plot the connection between the last x_train value and the first prediction
        last_train_date = commits_df.index[len(x_train) - 1]
        last_train_value = y_train[-1]
        first_prediction_value = predictions[0].item()
        first_prediction_date = prediction_dates[0]

        # Plot the line connecting the last train point to the first prediction point
        plt.plot([last_train_date, first_prediction_date], [last_train_value, first_prediction_value], color='red',
                 linestyle='--')

        plt.title(f'LSTMModel Predictions vs Actual Values ({task})')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        self.save_plot(f'lstm_predictions_{task}.png')

    def plot_commit_predictions(self, commits_df, model_info, task):
        plt.figure(figsize=(12, 6))
        #commits_df = commits_df[commits_df[task] > 0]

        plt.plot(commits_df.index, commits_df[task], label=f'Historical {task.capitalize()}', linestyle='-',
                color='blue')

        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_cycle = cycle(colors)

        for model_name, info in model_info.items():
            if info['x_test'] is None:
                # Use the time index from commits_df for ARIMA/SARIMA models
                prediction_start_index = commits_df.index[-len(info['y_test']):]
            else:
                prediction_start_index = pd.to_datetime(info['x_test']).tz_localize(None)

            predictions = info['predictions']
            predicted_df = pd.DataFrame({task: predictions.flatten()}, index=prediction_start_index)

            current_colour = next(color_cycle)

            plt.plot(predicted_df.index, predicted_df[task],
                    label=f'Prediction by {model_name} (MSE: {info["mse"]:.2f}, MAE: {info["mae"]:.2f}, '
                        f'RMSE: {info["rmse"]:.2f})', linestyle='-', color=current_colour)

            # Plot the last training point connected to the first prediction point
            last_train_date = commits_df.index[len(commits_df) - len(info['y_test']) - 1]
            last_train_value = commits_df[task].iloc[-len(info['y_test']) - 1]
            first_prediction_value = predictions.flatten()[0]

            plt.plot([last_train_date, prediction_start_index[0]],
                     [last_train_value, first_prediction_value], color=current_colour, linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'{task.capitalize()} Over Time with Forecast')
        plt.grid(True)
        plt.legend()

        self.save_plot(f'commit_predictions_{task}.png')

    def plot_predictions(self, filedata_df, model_info, label, target):
        plt.figure(figsize=(12, 6))

        # Plot actual data
        plt.plot(filedata_df.index, filedata_df[target], label=f"Actual {target.capitalize()}", color='blue', linestyle='solid')

        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_cycle = cycle(colors)

        for model_name, info in model_info.items():
            # Prepare data for the specific model
            if model_name == "ProphetModel":
                x_train_dates = pd.to_datetime(info['x_train'], errors='coerce').tz_localize(None)
                x_test_dates = pd.to_datetime(info['x_test'], errors='coerce').tz_localize(None)
                predictions = info['predictions']
            else:  # General handling (e.g., LSTM or others)
                x_train_dates = pd.to_datetime(info['x_train'], errors='coerce').tz_localize(None)
                x_test_dates = pd.to_datetime(info['x_test'], errors='coerce').tz_localize(None)
                predictions = (info['predictions'].values
                               if isinstance(info['predictions'], pd.Series)
                               else info['predictions'])

            # Plot predictions
            current_color = next(color_cycle)
            predicted_df = pd.DataFrame({target: predictions}, index=x_test_dates)
            plt.plot(predicted_df.index, predicted_df[target],
                     label=f'{model_name} Predictions (MSE: {info["mse"]:.2f}, MAE: {info["mae"]:.2f}, RMSE: {info["rmse"]:.2f})',
                     color=current_color)

            if model_name == "ProphetModel":
                last_train_value = filedata_df[target].iloc[len(filedata_df) - len(info['y_test']) - 1]
                first_prediction_value = predictions[0]
                plt.plot(
                    [x_train_dates[-1], x_test_dates[0]],
                    [last_train_value, first_prediction_value],
                    linestyle='--', color=current_color, label=f'{model_name} Transition'
                )

            # Handle LSTM-specific connection (last train point to first prediction)
            if model_name == "LSTMModel":
                last_train_value = info['y_train'][-1]
                first_prediction_value = predictions[0].item()
                plt.plot([x_train_dates[-1], x_test_dates[0]], [last_train_value, first_prediction_value],
                         linestyle='--', color=current_color)

        plt.title(f'{target.capitalize()} Over Time for {label}')
        plt.xlabel('Date')
        plt.ylabel(f'{target.capitalize()}')
        plt.legend()
        plt.grid(True)

        # Save plot with label in filename
        sanitized_label = label.replace("/", "_").replace(" ", "_")
        self.save_plot(f'predictions_{target}_{sanitized_label}.png')


    def plot_clusters(self, combined_df):
        """
        Creates scatter plot of all data points, coloured by clusters
        :param combined_df:
        :return:x
        """
        plt.figure(figsize=(10, 8))
        plt.scatter(
            combined_df['cooccurrence_scaled'],
            combined_df['distance_scaled'],
            c=combined_df['cluster'],
            cmap='viridis'
        )
        plt.xlabel('Co-occurrence (scaled)')
        plt.ylabel('Distance (scaled)')
        plt.title('File Pair Clustering by Co-occurrence and Distance')
        plt.colorbar(label='Cluster')
        self.save_plot('clusters.png')
