import logging
import os

import seaborn as sns
from matplotlib import pyplot as plt

from src.visualisations.plotting import Plotter


class ModelPlotter(Plotter):
    def __init__(self, project_name=None, images_dir='images'):
        super().__init__(project_name, images_dir)
        self.logging = logging.getLogger(self.__class__.__name__)

        self.images_dir = images_dir
        self.project_name = project_name

    def plot_residuals(self, y_true, y_pred):
        residuals = y_true - y_pred

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(residuals, kde=True)
        plt.title("Distribution of Residuals")
        plt.xlabel("Errors (y_true - y_pred)")

        plt.subplot(1, 2, 2)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.title("Predictions vs. Errors")
        plt.xlabel("Predicted Days Until Completion")
        plt.ylabel("Errors (y_true - y_pred)")
        plt.tight_layout()

        self.save_plot("residuals_model.png")

    def plot_errors_vs_actual(self, y_true, y_pred):
        errors = y_true - y_pred

        self._init_plot(title="Errors vs. Actual Days Until Completion", xlabel="Actual Days Until Completion",
                        ylabel="Errors (y_true - y_pred)")

        plt.scatter(y_true, errors, alpha=0.7)
        plt.axhline(0, color="red", linestyle="--")
        plt.tight_layout()

        self.save_plot("errors_vs_actual_model.png")

    def plot_completion_donut(self, completed, total):
        remaining = total - completed
        labels = ['Completed', 'Incomplete']
        sizes = [completed, remaining]

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, startangle=90, autopct='%1.1f%%', wedgeprops={'width': 0.3})
        ax.set_title('Completed Files')

        self.save_plot(f"completed_files_donut.png")