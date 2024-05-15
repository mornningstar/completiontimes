import os
from itertools import cycle

import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.predictions.machine_learning.decision_tree import DecisionTreeModel

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

    async def run(self):
        await self.fetch_data()
        model_info = self.train_and_evaluate_model()
        self.plot_data(model_info)

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
        """
        Prepares the data for visualisation.
        :param test_size: Percentage of data to be used for testing
        :return:
            x_train: timestamps of train datatest,
            x_test: timestamps of test dataset,
            y_train: file sizes of train dataset,
            y_test: file sizes of test dataset
        """
        self.size_df.sort_index(inplace=True)

        train_size = 1 - test_size
        train, test = train_test_split(self.size_df, train_size=train_size, shuffle=False)

        x_train = train.index.astype(int).values.reshape(-1, 1)
        y_train = train['size']

        x_test = test.index.astype(int).values.reshape(-1, 1)
        y_test = test['size']

        return x_train, y_train, x_test, y_test

    def train_and_evaluate_model(self):
        x_train, y_train, x_test, y_test = self.prepare_data()
        model_info = {}

        for model in self.models:
            # Only call auto_tune() when the model class has this method
            if hasattr(model, 'auto_tune'):
                model.auto_tune(y_train)

            model.train(x_train, y_train)
            predictions, mse = model.evaluate(y_test, x_test)

            model_info[model.__class__.__name__] = {
                'mse': mse,
                'predictions': predictions,
                'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test
            }
            print(f"Trained {model.__class__.__name__} with MSE: {mse}")
            print(f"Predictions: {predictions}")

        return model_info

    def plot_data(self, model_info):
        if self.size_df is None:
            raise ValueError('Data is not processed. Call process_data() before plotting!')

        images_dir = 'images'
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        plt.figure(figsize=(12, 6))

        # Plot historical data
        plt.plot(self.size_df.index, self.size_df['size'], label='Historical File Size', linestyle='-',
                 marker='o',color='blue')

        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_cycle = cycle(colors)

        for model_name, info in model_info.items():
            x_train_dates = pd.to_datetime(info['x_train'].flatten())
            x_test_dates = pd.to_datetime(info['x_test'].flatten())

            last_train_point = info['y_train'].iloc[-1]  # Get last training point size
            last_train_date = x_train_dates[-1]  # Get last training point date

            if isinstance(info['predictions'], pd.Series):
                predictions = info['predictions'].values
            else:
                predictions = info['predictions']

            predicted_df = pd.DataFrame({
                'size': predictions#info['predictions'].values,
            }, index=x_test_dates)

            current_colour = next(color_cycle)

            # Plot predicted data
            plt.plot(predicted_df.index, predicted_df['size'],
                     label=f'Prediction by {model_name} (MSE: {info["mse"]:.2f})', linestyle='--', marker='o',
                     color=current_colour)

            if not predicted_df.empty:
                first_pred_size = predicted_df.iloc[0]['size']

                # Plot a line between the last training point and the first predicted point
                plt.plot([last_train_date, x_test_dates[0]], [last_train_point, first_pred_size],
                         color=current_colour, linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('File Size')
        plt.title(f'File Size Over Time for {self.file_path}')
        plt.grid(True)
        plt.legend()

        plt.savefig(f'{images_dir}/plot_{self.file_path.replace("/", "_")}.png')
