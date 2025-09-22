import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .metric_protocol import Metric

class MAEMetric(Metric):
    def name(self):
        return "MAE"

    def compute(self, y_true, y_pred):
        return mean_absolute_error(y_true=y_true, y_pred=y_pred)


class MSEMetric(Metric):
    def name(self):
        return "MSE"

    def compute(self, y_true, y_pred):
        return mean_squared_error(y_true=y_true, y_pred=y_pred)

class RMSEMetric(Metric):
    def name(self):
        return "RMSE"

    def compute(self, y_true, y_pred):
        return np.sqrt(MSEMetric.compute(y_true, y_pred))