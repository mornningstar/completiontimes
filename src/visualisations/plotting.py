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

    @staticmethod
    def _create_directory(directory_path):
        """Helper function to create a directory if it does not exist."""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    @staticmethod
    def _init_plot(title=None, xlabel=None, ylabel=None, figsize=(12, 6)):
        plt.figure(figsize=figsize)
        if title: plt.title(title)
        if xlabel: plt.xlabel(xlabel)
        if ylabel: plt.ylabel(ylabel)

        plt.grid(True)

    def save_plot(self, filename):
        """Helper function to save the current plot to the project directory."""
        plt.savefig(os.path.join(self.images_dir, filename))
        plt.close()

    # def plot(self, df, title, ylabel):
    #     """
    #     Helper function to plot the given dataframe.
    #     :param df: dataframe to plot
    #     :param title: title of the plot
    #     :param ylabel: Label for the y-axis
    #     :return:
    #     """
    #     self._init_plot(title, xlabel="Time", ylabel=ylabel)
    #
    #     plt.plot(df.index, df.iloc[:, 0], marker='o', linestyle='-')
    #
    #     self.save_plot("plot.png")


