from src.predictions.machine_learning.decision_tree import DecisionTreeModel
from src.predictions.machine_learning.lstmmodel import LSTMModel
from src.predictions.prophet_model import ProphetModel
from src.predictions.statistical_predictions.exponential_smoothing import ExponentialSmoothingModel
from src.predictions.statistical_predictions.seasonal_arima_base import SeasonalARIMABase

PROJECTS = [
    {
        'name': 'khoj-ai/khoj',
        'modeling': [
            #'rolling_7_commit_count',
            #'rolling_7_additions',
            #'rolling_7_deletions',
            #'commit_rate_ema',
            #'cumulative_additions',
            #'cumulative_deletions',
            #'cumulative_net_changes',
            #'additions_to_deletions_ratio',
            #'lag_7_commit_count',

        ],
        'models': [
            ProphetModel,
            #SeasonalARIMABase,
            #LSTMModel(),
            #ARIMAModel(),
            #SARIMAModel(),
            #ExponentialSmoothingModel(trend='add', seasonal='add', seasonal_periods=7),
            #DecisionTreeModel(grid_search=True),
        ],
        'file_paths': [
            'src/khoj/routers/web_client.py'
        ],
        'file_modeling_tasks': [
            'rolling_7_size',
            'rolling_7_std',
            'rolling_7_max',
            'rolling_7_min',
            'rolling_7_median',
            'rolling_7_var',
            'size_ema',
            'cumulative_size',
            'cumulative_mean',
            'cumulative_std',
            'lag_1_size',
            'lag_2_size',
            'lag_3_size',
            'lag_4_size',
            'lag_5_size',
            'lag_6_size',
            'lag_7_size',
            'absolute_change',
            'percentage_change',
            'rolling_mean_to_std_ratio'
            #'size',
            #'rolling_7_std',
        ]
    },
    #{
        #'name': 'twentyhq/twenty',
        #'modeling': [
            #'totals',
        #],
        #'models': [],
        #'file_paths': []
    #}
]
