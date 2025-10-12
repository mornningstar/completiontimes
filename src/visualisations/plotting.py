import logging
import os

import seaborn as sns

from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, project_name=None, images_dir='images'):
        self.logging = logging.getLogger(self.__class__.__name__)
        self.images_dir = images_dir
        self.project_name = project_name

        self._create_directory(self.images_dir)

    @staticmethod
    def _create_directory(directory_path):
        """Helper function to create a directory if it does not exist."""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    @staticmethod
    def _init_plot(title=None, xlabel=None, ylabel=None, figsize=(12, 6)):
        plt.figure(figsize=figsize)
        plt.tight_layout()

        if title: plt.title(title)
        if xlabel: plt.xlabel(xlabel)
        if ylabel: plt.ylabel(ylabel)

        plt.grid(True)

    def plot_bar(self, series, title, xlabel, ylabel, rotation=45, filename=None, yerr=None):
        self._init_plot(title=title, xlabel=xlabel, ylabel=ylabel, figsize=(10, 6))

        series.plot(kind='bar', rot=rotation, yerr=yerr)

        if filename is None:
            filename = f"{title.lower().replace(' ', '_')}.png"

        self.save_plot(filename)

    def save_plot(self, filename):
        """Helper function to save the current plot to the project directory."""
        plt.savefig(os.path.join(self.images_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_violin(self, data, x, y, title, xlabel, ylabel, filename=None):
        self._init_plot(title=title, xlabel=xlabel, ylabel=ylabel)
        sns.violinplot(x=x, y=y, data=data)

        if filename is None:
            filename = f"{title.lower().replace(' ', '_')}.png"
        self.save_plot(filename)
