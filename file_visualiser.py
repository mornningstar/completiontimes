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
        next_size, next_date = predictor.predict_next_size()  # possible next values

        # Extend historical data with the predicted point for plotting
        extended_size_df = self.size_df.concat(self.size_df, pd.DataFrame({
            'time': next_date,
            'size': next_size
        }))

        plt.figure(figsize=(12, 6))
        # plt.plot(self.size_df.index, self.size_df['size'], label='Historical File Size', color='blue')

        # Plot predicted data
        # plt.plot(next_date, next_size, label='Predicted File Size', color='red', linestyle='--', marker='o')

        # Plot historical data
        plt.plot(self.size_df.index, self.size_df['size'], label='Historical File Size', color='blue')
        # Plot extended data with prediction to draw the connecting line
        plt.plot(extended_size_df.index, extended_size_df, label='Prediction', color='red', linestyle='--', marker='o')

        plt.xlabel('Date')
        plt.ylabel('File Size')
        plt.title(f'File Size Over Time for {self.file_path}')
        plt.grid(True)
        plt.legend()

        plt.savefig(f'images/plot_{self.file_path.replace("/", "_")}.png')
