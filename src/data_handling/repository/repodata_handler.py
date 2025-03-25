import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller

from src.data_handling.database.async_database import AsyncDatabase
from src.data_handling.repository.repository_size_handling import RepoSizeHandler
from src.predictions.model_training import ModelTrainer
from src.visualisations.plotting import Plotter


def add_features(dataframe, time_col='time', count_col='commits', additions_col='additions',
                 deletions_col='deletions', window=7):

    # A moving average of commit counts
    dataframe[f'rolling_{window}_commit_count'] = dataframe[count_col].rolling(window=window).mean()

    # A moving average for commit additions
    dataframe[f'rolling_{window}_additions'] = dataframe[additions_col].rolling(window=window).mean()

    # A moving average for commit deletions
    dataframe[f'rolling_{window}_deletions'] = dataframe[deletions_col].rolling(window=window).mean()

    # Exponential moving average of commit rate
    dataframe['commit_rate_ema'] = dataframe[count_col].ewm(span=window, adjust=False).mean()

    # Cumulative changes
    dataframe['cumulative_additions'] = dataframe[additions_col].cumsum()
    dataframe['cumulative_deletions'] = dataframe[deletions_col].cumsum()
    dataframe['cumulative_net_changes'] = dataframe['cumulative_additions'] - dataframe['cumulative_deletions']

    dataframe['additions_to_deletions_ratio'] = dataframe[additions_col] / dataframe[deletions_col].replace(0, 1)

    dataframe[f'lag_{window}_commit_count'] = dataframe[count_col].shift(window)
    dataframe[f'lag_{window}_additions'] = dataframe[additions_col].shift(window)
    dataframe[f'lag_{window}_deletions'] = dataframe[deletions_col].shift(window)

    return dataframe


class RepoDataHandler:
    def __init__(self, api_connection, models, modeling_tasks):
        self.api_connection = api_connection
        self.model_trainer = ModelTrainer(models, modeling_tasks=modeling_tasks)
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
        df.sort_index(inplace=True)

        df = df.groupby(df.index.date).sum()
        df.index = pd.to_datetime(df.index)

        self.commits_df = df.resample('W').asfreq().fillna(0)

        #self.commits_df.ffill(inplace=True)
        #self.commits_df.bfill(inplace=True)

        #self.commits_df = self.commits_df[self.commits_df['commits'] > 0]

        self.commits_df.index = self.commits_df.index.tz_localize(None)

        self.commits_df = add_features(self.commits_df, time_col='time', count_col='commits',
                                       additions_col='additions', deletions_col='deletions')

    def prepare_arima_data(self, test_size=0.2):
        # Ensure the DataFrame is sorted by the time index
        self.commits_df.sort_index(inplace=True)

        data_splits = {}

        for task in self.modeling_tasks:
            if task in self.commits_df.columns:
                arima_ready_df = self.commits_df[[task]].copy()

                arima_ready_df.dropna(inplace=True)
                #arima_ready_df.fillna(0, inplace=True)

                train_size = int((1 - test_size) * len(arima_ready_df))
                train = arima_ready_df[:train_size]
                test = arima_ready_df[train_size:]

                # On the y-axis are the target values (the task values)
                y_train = train[task].values
                y_test = test[task].values

                # time (i.e. the x-axis) is implicit via the index
                data_splits[task] = (None, y_train, None, y_test)

            else:
                raise ValueError(f"Task '{task}' is not a valid column in the data.")

        return data_splits

    def prepare_lstm_data(self, timesteps=10, test_size=0.2):
        self.commits_df.sort_index(inplace=True)
        self.commits_df.ffill(inplace=True)
        self.commits_df.bfill(inplace=True)

        task_datasets = {}

        for task in self.modeling_tasks:
            if task in self.commits_df.columns:
                # Split into training and test sets for this specific task (target variable)
                train_size = int((1 - test_size) * len(self.commits_df))
                df_train = self.commits_df[:train_size]
                df_test = self.commits_df[train_size:]

                # Prepare the features (all columns except the target)
                train_features = df_train.drop(columns=[task]).values
                test_features = df_test.drop(columns=[task]).values

                # Prepare the target variable (the specific column for this task)
                y_train = df_train[task].values[timesteps:]
                y_test = df_test[task].values[timesteps:]

                # Prepare LSTM-specific 3D data [samples, timesteps, features]
                x_train = np.array([train_features[i:i + timesteps] for i in range(len(df_train) - timesteps)])
                x_test = np.array([test_features[i:i + timesteps] for i in range(len(df_test) - timesteps)])

                task_datasets[task] = (x_train, y_train, x_test, y_test)
            else:
                raise ValueError(f"Task '{task}' not found in DataFrame columns")

        return task_datasets

    async def prepare_data(self, test_size=0.2):
        self.commits_df.sort_index(inplace=True)

        data_splits = {}

        if 'repo_size' in self.modeling_tasks:
            reposize_handler = RepoSizeHandler(self.api_connection.file_tracking_collection)
            await reposize_handler.fetch_repository_sizes(self.commit_data)

            self.commits_df = self.commits_df.merge(
                reposize_handler.repo_size_df, how='left', left_index=True, right_index=True
            )

        #self.commits_df.ffill(inplace=True)
        #self.commits_df.bfill(inplace=True)

        for task in self.modeling_tasks:
            if task in self.commits_df.columns:
                train_size = 1 - test_size
                train, test = train_test_split(self.commits_df[task], train_size=train_size, shuffle=False)

                # Convert index to integer timestamps for model compatibility
                x_train = train.index
                y_train = train.values

                x_test = test.index
                y_test = test.values

                data_splits[task] = (x_train, y_train, x_test, y_test)

                print(len(x_train), len(y_train), len(x_test), len(y_test))

            else:
                raise ValueError(f"Task '{task}' is not a valid column in the data.")

        return data_splits

    def check_and_differencing(self, dataframe, task, threshold=0.05):
        # Perform Dickey-Fuller test to check stationarity
        result = adfuller(dataframe[task].dropna())
        p_value = result[1]

        if p_value > threshold:
            print(f"{task} is not stationary. Applying differencing...")
            dataframe[task] = dataframe[task].diff().dropna()  # Apply differencing if not stationary
        else:
            print(f"{task} is stationary.")

    async def plot(self):
        commit_stats_to_plot = [task for task in self.modeling_tasks if task != 'repo_size']

        if 'repo_size' in self.modeling_tasks:
            reposize_handler = RepoSizeHandler(self.api_connection.file_tracking_collection)
            await reposize_handler.fetch_repository_sizes(self.commit_data)

            reposize_handler.plot_repository_size()

        if commit_stats_to_plot:
            self.plotter.plot_commits(self.commits_df, commit_stats_to_plot)
