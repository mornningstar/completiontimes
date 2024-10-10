from src.predictions.machine_learning.decision_tree import DecisionTreeModel
from src.predictions.statistical_predictions.arima import ARIMAModel
from src.predictions.statistical_predictions.exponential_smoothing import SimpleExponentialSmoothing
from src.predictions.statistical_predictions.sarima import SARIMAModel

PROJECTS = [
    {
        'name': 'khoj-ai/khoj',
        'modeling': [
            'totals',
            'cumulative_net_changes'
        ],
        'models': [
            ARIMAModel(),
            SARIMAModel()
            #SimpleExponentialSmoothing(),
            #DecisionTreeModel(max_depth=1),
        ],
        'file_paths': [
            #'src/khoj/routers/web_client.py'
        ]
    },
]

