from src.predictions.machine_learning.decision_tree import DecisionTreeModel
from src.predictions.statistical_predictions.arima import ARIMAModel
from src.predictions.statistical_predictions.exponential_smoothing import SimpleExponentialSmoothing

PROJECTS = [
    {
        'name': 'ohmyzsh/ohmyzsh',
        'commit_views': [
            'totals',
            'additions',
            'deletions',
        ],
        'models': [
            ARIMAModel(),
            SimpleExponentialSmoothing(),
            DecisionTreeModel(max_depth=1),
        ],
        'file_paths': [
            'plugins/heroku/_heroku',
            'plugins/ubuntu/ubuntu.plugin.zsh',
            'plugins/mercurial/mercurial.plugin.zsh',
        ]
    },
]





