import pandas as pd

from src.data_handling.database.async_database import AsyncDatabase
from src.visualisations.plotting import Plotter


class RepoSizeHandler:
    def __init__(self, collection_name):
        self.collection_name = collection_name

        self.plotter = Plotter(self.collection_name)

        self.repo_size_df = None

    async def fetch_repository_sizes(self, commit_data):
        print("Fetching repository sizes...")
        file_data = await AsyncDatabase.find(self.collection_name, {})
        file_commit_map = {}

        for file_entry in file_data:
            file_path = file_entry['path']
            if 'commit_history' in file_entry:
                file_commit_map[file_path] = file_entry['commit_history']

        repo_sizes = []
        current_file_sizes = {}

        # Sort commits by date to process them chronologically
        commit_data.sort(key=lambda x: pd.to_datetime(x['commit']['author']['date']))

        for commit in commit_data:
            commit_sha = commit['sha']
            commit_date = pd.to_datetime(commit['commit']['author']['date'])

            # Update the current file sizes based on the commit
            for file_path, commit_history in file_commit_map.items():
                for file_commit in commit_history:
                    if file_commit['sha'] == commit_sha:
                        size = file_commit.get('size', None)
                        if size == 'file_not_found':
                            #current_file_sizes[file_path] = 0
                            current_file_sizes.pop(file_path, None)
                        else:
                            current_file_sizes[file_path] = size
                        break

            # Calculate the total repository size by summing up the current sizes of all files
            repo_size = sum(current_file_sizes.values())
            repo_sizes.append({'time': commit_date, 'repo_size': repo_size})

        repo_size_df = pd.DataFrame(repo_sizes)#.set_index('time').resample('D').ffill()
        repo_size_df = repo_size_df.groupby('time').sum()

        # Make dataframe time-zone naive
        repo_size_df.index = repo_size_df.index.tz_convert(None)

        self.repo_size_df = repo_size_df.resample('D').ffill()
        self.repo_size_df = self.repo_size_df[self.repo_size_df['repo_size'] > 0]

    def plot_repository_size(self):
        self.plotter.plot(self.repo_size_df, title="Repository Size Over Time", ylabel="Size (bytes)")
