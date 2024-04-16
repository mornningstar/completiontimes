import os

import matplotlib
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from async_database import AsyncDatabase
from file_size_predictor import FileSizePredictor


class FileVisualiser:

    def __init__(self, collection_name, file_path):
        self.collection_name = collection_name
        self.file_path = file_path
        self.file_data = None
        self.size_df = None

    async def fetch_data(self):
        query = {"path": self.file_path}
        self.file_data = await AsyncDatabase.find(self.collection_name, query)

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
        })

        df.sort_values(by='time', inplace=True)
        df.set_index('time', inplace=True)
        self.size_df = df

    def plot_data(self):
        if self.size_df is None:
            raise ValueError('Data is not processed. Call process_data() before plotting!')

        images_dir = 'images'
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        print(self.size_df)
        print(self.size_df.dtypes)

        predictor = FileSizePredictor(self.size_df)
        predictor.prepare_data()
        predictor.train_model()
        predicted_sizes, predicted_dates = predictor.predict_next_sizes()

        predicted_df = pd.DataFrame({
            'size': predicted_sizes
        }, index=predicted_dates)

        # Extend historical data with the predicted point for plotting
        extended_size_df = pd.concat([
            self.size_df,
            predicted_df
        ])

        plt.figure(figsize=(12, 6))

        # Plot historical data
        plt.plot(self.size_df.index, self.size_df['size'], label='Historical File Size', color='blue')

        # Plot predicted data
        plt.plot(predicted_df.index, predicted_df, label='Prediction', color='red', linestyle='--', marker='o')

        # Plot line from real to predicted data
        plt.plot([self.size_df.index[-1], predicted_dates[0]], [self.size_df['size'].iloc[-1], predicted_sizes[0]],
                 color='red', linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('File Size')
        plt.title(f'File Size Over Time for {self.file_path}')
        plt.grid(True)
        plt.legend()

        plt.savefig(f'images/plot_{self.file_path.replace("/", "_")}.png')
