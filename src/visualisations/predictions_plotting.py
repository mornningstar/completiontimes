from itertools import cycle

import pandas as pd
from matplotlib import pyplot as plt

from src.visualisations.plotting import Plotter


class PredictionsPlotter(Plotter):
    def __init__(self, project_name=None, images_dir='images'):
        super().__init__(project_name, images_dir)


    def plot_commits(self, data, stats_to_plot):
        self._init_plot(title="Changes Over Time", xlabel="Date", ylabel="Count")

        for stat in stats_to_plot:
            if stat in data.columns:
                plt.plot(data.index, data[stat], label=stat.capitalize())
            else:
                raise ValueError(f'Stat {stat} does not exist in the data.')

        plt.legend()

        super().save_plot("commits.png")


    def plot_lstm_predictions(self, commits_df, model_info, task):
        self._init_plot(title=f"LSTMModel Predictions vs Actual Values ({task})", xlabel="Date", ylabel="Prediction Value")

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

        plt.legend()

        self.save_plot(f'lstm_predictions_{task}.png')


    def plot_commit_predictions(self, commits_df, model_info, task):
        self._init_plot(title=f"{task.capitalize()} Over Time with Forecast", xlabel="Date", ylabel="Value")

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

        plt.legend()

        self.save_plot(f'commit_predictions_{task}.png')


    def plot_predictions(self, filedata_df, model_info, label, target):
        self._init_plot(title=f"{target.capitalize()} Over Time for {label}", xlabel='Date',
                    ylabel=f"{target.capitalize()}")

        # Plot actual data
        plt.plot(filedata_df.index, filedata_df[target], label=f"Actual {target.capitalize()}", color='blue',
             linestyle='solid')

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

        plt.legend()

        # Save plot with label in filename
        sanitized_label = label.replace("/", "_").replace(" ", "_")
        self.save_plot(f'predictions_{target}_{sanitized_label}.png')


    def plot_clusters(self, combined_df):
        """
        Creates scatter plot of all data points, coloured by clusters
        :param self:
        :param combined_df:
        :return:x
        """
        self._init_plot(title="File Pair Clustering by Co-occurrence and Distance", xlabel="Co-occurrence (scaled)",
                    ylabel="Distance (scaled)", figsize=(10, 8))
        plt.scatter(
            combined_df['cooccurrence_scaled'],
            combined_df['distance_scaled'],
            c=combined_df['cluster'],
            cmap='viridis'
        )
        plt.colorbar(label='Cluster')
        self.save_plot('clusters.png')


    def plot_refit_predictions(self, historical_series, future_dates, future_values, label, target):
        """
        Plots the historical data plus the new refit forecast.
        """
        self._init_plot(title=f"Refit Forecast for {label} {target}", xlabel="Date", ylabel=target)

        plt.plot(historical_series.index, historical_series.values, label='Historical Data')
        plt.plot(future_dates, future_values, 'r--', label=f'Refit Forecast ({label})')
        plt.legend()

        self.save_plot(f"{label}_{target}_refit.png")