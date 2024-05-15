import os
from itertools import cycle

import pandas as pd
from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, images_dir='images'):
        self.images_dir = images_dir

        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

    def plot(self, size_df, model_info, file_path):
        plt.figure(figsize=(12, 6))

        plt.plot(size_df.index, size_df['size'], label='Historical File Size', linestyle='-', marker='o', color='blue')

        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_cycle = cycle(colors)

        for model_name, info in model_info.items():
            x_train_dates = pd.to_datetime(info['x_train'].flatten())
            x_test_dates = pd.to_datetime(info['x_test'].flatten())

            last_train_point = info['y_train'].iloc[-1]
            last_train_date = x_train_dates[-1]

            if isinstance(info['predictions'], pd.Series):
                predictions = info['predictions'].values
            else:
                predictions = info['predictions']

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

        plt.savefig(f'{self.images_dir}/plot_{file_path.replace("/", "_")}.png')
