import os

from src.data_handling.async_database import AsyncDatabase

import pandas as pd
import matplotlib.pyplot as plt


class CommitVisualiser:

    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.commit_data = None
        self.daily_df = None
        self.daily_commit_count_df = None

    async def fetch_data(self):
        self.commit_data = await AsyncDatabase.find(self.collection_name, {})

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
            'commit_counts': commit_counts
        })

        df.sort_values(by='time', inplace=True)
        df.set_index('time', inplace=True)
        self.daily_df = df.resample('D').sum()
        self.daily_commit_count_df = df.resample('D').count()['commit_counts']

    def plot_data(self, stats_to_plot):
        if self.daily_df is None or self.daily_commit_count_df is None:
            raise ValueError('Data is not processed. Call process_data() before plotting!')

        images_dir = '../../images'
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        plt.figure(figsize=(12, 6))

        for stat in stats_to_plot:
            if stat == 'commits':
                # Plot the commits separately since it's not in daily_df
                plt.plot(self.daily_commit_count_df.index, self.daily_commit_count_df, label='Commits')
            elif stat in self.daily_df.columns:
                # Plot other stats from daily_df
                plt.plot(self.daily_df.index, self.daily_df[stat], label=stat.capitalize())
            else:
                # Raise an error if a stat is neither 'commits' nor in daily_df
                raise ValueError(f'Stat {stat} does not exist in the data.')

        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.title('Changes Over Time')
        plt.grid(True)
        plt.legend()

        plt.savefig(f'images/plot_{self.collection_name.replace("/", "_")}.png')
