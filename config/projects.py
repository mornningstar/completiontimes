from src.predictions.machine_learning.decision_tree import DecisionTreeModel
from src.predictions.machine_learning.lstmmodel import LSTMModel
from src.predictions.prophet_model import ProphetModel
from src.predictions.statistical_predictions.exponential_smoothing import ExponentialSmoothingModel
from src.predictions.statistical_predictions.seasonal_arima_base import SeasonalARIMABase

PROJECTS = [
    {
        'name': 'mozilla/addons-server',
    },
    {
        'name': 'openedx/edx-platform'
    }
]
