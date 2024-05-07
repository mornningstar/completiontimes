import os
from itertools import cycle

import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from src.data_handling.async_database import AsyncDatabase


class FileVisualiser:

    def __init__(self, collection_name, file_path, models):
        self.y_test = None
        self.collection_name = collection_name
        self.file_path = file_path
        self.file_data = None
        self.size_df = None
        self.models = models

        images_dir = '../../images'
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

    async def fetch_data(self):
        query = {"path": self.file_path}
        self.file_data = await AsyncDatabase.find(self.collection_name, query)
        self.process_data()

    def process_data(self):
        times = []
        sizes = []

        for record in self.file_data:
            for commit in record.get('commit_history', []):
                commit_date = pd.to_datetime(commit['date'])
                file_size = commit.get('size', 0)

                times.append(commit_date)
                sizes.append(file_size)

        df = pd.DataFrame({
            'time': times,
            'size': sizes
        }).set_index('time').sort_values('time')

        #df.sort_values(by='time', inplace=True)
        #df.set_index('time', inplace=True)
        df.index = pd.DatetimeIndex(df.index)

        self.size_df = df

    def prepare_data(self, test_size=0.2):
        self.size_df.sort_index(inplace=True)

        train_size = 1 - test_size
        X_train, X_test = train_test_split(self.size_df, train_size=train_size, shuffle=False)
        return X_train, X_test

    def train_and_evaluate_model(self):
        X_train, X_test = self.prepare_data()

        model_info = {}
        for model in self.models:
            # Only call auto_tune() when the model class has this method
            if hasattr(model, 'auto_tune'):
                model.auto_tune(X_train['size'])

            model.train(X_train['size'])
            predictions, mse = model.evaluate(X_test['size'])

            model_info[model.__class__.__name__] = {
                'mse': mse,
                'predictions': predictions,
                'X_test': X_test,
                'X_train': X_train

            }
            print(f"Trained {model.__class__.__name__} with MSE: {mse}")

        return model_info

    def plot_data(self, model_info):
        if self.size_df is None:
            raise ValueError('Data is not processed. Call process_data() before plotting!')

        plt.figure(figsize=(12, 6))

        # Plot historical data
        plt.plot(self.size_df.index, self.size_df['size'], label='Historical File Size', linestyle='-',
                 marker='o',color='blue')

        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_cycle = cycle(colors)

        for model_name, info in model_info.items():
            last_train_point = info['X_train']['size'].iloc[-1]  # Get last training point size
            last_train_date = info['X_train'].index[-1]  # Get last training point date

            prediction_dates = info['X_test'].index

            predicted_df = pd.DataFrame({
                'size': info['predictions'].values,
            }, index=prediction_dates)

            current_colour = next(color_cycle)

            # Plot predicted data
            plt.plot(predicted_df.index, predicted_df['size'],
                     label=f'Prediction by {model_name} (MSE: {info["mse"]:.2f})', linestyle='--', marker='o',
                     color=current_colour)

            if not predicted_df.empty:
                first_pred_size = predicted_df.iloc[0]['size']
                # Ensure you plot a line between the last training point and the first predicted point
                plt.plot([last_train_date, prediction_dates[0]], [last_train_point, first_pred_size],
                         color=current_colour, linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('File Size')
        plt.title(f'File Size Over Time for {self.file_path}')
        plt.grid(True)
        plt.legend()

        plt.savefig(f'images/plot_{self.file_path.replace("/", "_")}.png')
