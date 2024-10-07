import pandas as pd

from src.data_handling.async_database import AsyncDatabase
from src.data_handling.repository_size_handling import RepoSizeHandler
from src.visualisations.plotting import Plotter


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

    async def plot(self):
        commit_stats_to_plot = [task for task in self.modeling_tasks if task != 'repo_size']

        if 'repo_size' in self.modeling_tasks:
            reposize_handler = RepoSizeHandler(self.api_connection.file_tracking_collection)
            await reposize_handler.fetch_repository_sizes(self.commit_data)
            reposize_handler.plot_repository_size()

        if commit_stats_to_plot:
            self.plotter.plot_commits(self.commits_df, commit_stats_to_plot)
