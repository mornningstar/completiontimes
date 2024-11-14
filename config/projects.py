from src.predictions.machine_learning.decision_tree import DecisionTreeModel
from src.predictions.machine_learning.lstmmodel import LSTMModel
from src.predictions.prophet_model import ProphetModel
from src.predictions.statistical_predictions.arima import ARIMAModel
from src.predictions.statistical_predictions.exponential_smoothing import ExponentialSmoothingModel
from src.predictions.statistical_predictions.sarima import SARIMAModel

PROJECTS = [
    {
        'name': 'khoj-ai/khoj',
        'modeling': [
            'rolling_7_commit_count',
            'rolling_7_additions',
            'rolling_7_deletions',
            'commit_rate_ema',
            'cumulative_additions',
            'cumulative_deletions',
            'cumulative_net_changes',
            'additions_to_deletions_ratio',
            'lag_7_commit_count',

        ],
        'models': [
            #ProphetModel,
            ARIMAModel,
            SARIMAModel,
            #LSTMModel(),
            #ARIMAModel(),
            #SARIMAModel(),
            #ExponentialSmoothingModel(trend='add', seasonal='add', seasonal_periods=7),
            #DecisionTreeModel(grid_search=True),
        ],
        'file_paths': [
            #'src/khoj/routers/web_client.py'
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
