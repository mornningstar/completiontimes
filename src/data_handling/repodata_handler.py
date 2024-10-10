import pandas as pd

from src.data_handling.async_database import AsyncDatabase
from src.data_handling.repository_size_handling import RepoSizeHandler
from src.visualisations.plotting import Plotter


def add_features(dataframe, time_col='time', count_col='commits', additions_col='additions',
                 deletions_col='deletions', window=7):

    dataframe[time_col] = pd.to_datetime(dataframe[time_col])
    dataframe.set_index(time_col, inplace=True)

    # A moving average of commit counts
    dataframe[f'rolling_{window}_commit_count'] = dataframe[count_col].rolling(window=window).mean()

    # A moving average for commit additions
    dataframe[f'rolling_{window}_additions'] = dataframe[additions_col].rolling(window=window).mean()

    # A moving average for commit deletions
    dataframe[f'rolling_{window}_deletions'] = dataframe[deletions_col].rolling(window=window).mean()

    # Cumulative changes
    dataframe['cumulative_additions'] = dataframe[additions_col].cumsum()
    dataframe['cumulative_deletions'] = dataframe[deletions_col].cumsum()
    dataframe['cumulative_net_changes'] = dataframe['cumulative_additions'] - dataframe['cumulative_deletions']

    dataframe['additions_to_deletions_ratio'] = dataframe[additions_col] / dataframe[deletions_col].replace(0, 1)

    dataframe['commit_rate'] = dataframe[count_col].rolling(window=window).sum() / window

    dataframe[f'lag_{window}_commit_count'] = dataframe[count_col].shift(window)
    dataframe[f'lag_{window}_additions'] = dataframe[additions_col].shift(window)
    dataframe[f'lag_{window}_deletions'] = dataframe[deletions_col].shift(window)

    dataframe['day_of_week'] = dataframe.index.day_of_week
    dataframe['month_of_year'] = dataframe.index.month

    return dataframe


class RepoDataHandler:
    def __init__(self, api_connection, modeling_tasks):
        self.api_connection = api_connection
        self.modeling_tasks = modeling_tasks

        # Currently only used for other commit stats
        self.plotter = Plotter(project_name=self.api_connection.full_commit_info_collection)

        self.commit_data = None
        self.commits_df = None

    async def run(self):
        await self.fetch_data()
        self.process_data()

    async def fetch_data(self):
        self.commit_data = await AsyncDatabase.find(self.api_connection.full_commit_info_collection, {})

    def process_data(self):
        times = []
        totals = []
        additions = []
        deletions = []
        commit_counts = []

        for commit in self.commit_data:
            try:
                commit_date = commit['commit']['author']['date']
                stats = commit['stats']

                commit_date = pd.to_datetime(commit_date)

                times.append(commit_date)
                totals.append(stats.get('total', 0))
                additions.append(stats.get('additions', 0))
                deletions.append(stats.get('deletions', 0))
                commit_counts.append(1)

            except KeyError:
                pass

        df = pd.DataFrame({
            'time': times,
            'totals': totals,
            'additions': additions,
            'deletions': deletions,
            'commits': commit_counts
        })

        df.set_index('time', inplace=True)
        df.index = pd.DatetimeIndex(df.index)
        df.sort_values(by='time', inplace=True)
        self.commits_df = df.resample('D').sum()

        self.commits_df = add_features(self.commits_df, time_col='time', count_col='commits',
                                       additions_col='additions', deletions_col='deletions')

    async def plot(self):
        commit_stats_to_plot = [task for task in self.modeling_tasks if task != 'repo_size']

        if 'repo_size' in self.modeling_tasks:
            reposize_handler = RepoSizeHandler(self.api_connection.file_tracking_collection)
            await reposize_handler.fetch_repository_sizes(self.commit_data)
            reposize_handler.plot_repository_size()

        if commit_stats_to_plot:
            self.plotter.plot_commits(self.commits_df, commit_stats_to_plot)
