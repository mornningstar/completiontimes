import logging
import os

import seaborn as sns
from matplotlib import pyplot as plt

from src.visualisations.plotting import Plotter


class ModelPlotter:
    def __init__(self, project_name=None, images_dir='images'):
        self.logging = logging.getLogger(self.__class__.__name__)
        self.images_dir = images_dir
        self.project_name = project_name

        self.plotter = Plotter(project_name=self.project_name)

        os.makedirs(images_dir, exist_ok=True)

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

        self.plotter.save_plot("residuals_model.png")

    def plot_errors_vs_actual(self, y_true, y_pred):
        errors = y_true - y_pred

        plt.figure(figsize=(12, 6))
        plt.scatter(y_true, errors, alpha=0.7)
        plt.axhline(0, color="red", linestyle="--")
        plt.title("Errors vs. Actual Days Until Completion")
        plt.xlabel("Actual Days Until Completion (y_true)")
        plt.ylabel("Errors (y_true - y_pred)")
        plt.grid(True)
        plt.tight_layout()

        self.plotter.save_plot("errors_vs_actual_model.png")