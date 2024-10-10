from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.predictions.base_model import BaseModel
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing


class ExponentialSmoothingModel(BaseModel):
    def __init__(self, trend=None, seasonal=None, seasonal_periods=None):
        super().__init__()
        self.trend = trend  # 'add' for additive trend, 'mul' for multiplicative
        self.seasonal = seasonal  # 'add' for additive seasonality, 'mul' for multiplicative
        self.seasonal_periods = seasonal_periods  # Number of periods in a full season cycle (e.g., 12 for monthly data if yearly seasonality)

    def train(self, x_train, y_train):
        self.model = self.model = ExponentialSmoothing(y_train,
                                                       trend=self.trend,
                                                       seasonal=self.seasonal,
                                                       seasonal_periods=self.seasonal_periods,
                                                       initialization_method='estimated').fit()

    def predict(self, steps):
        return self.model.forecast(steps=steps)

    def evaluate(self, y_test, x_test):
        predictions = self.predict(len(x_test))

        mse = mean_squared_error(y_true=y_test, y_pred=predictions)
        mae = mean_absolute_error(y_true=y_test, y_pred=predictions)

        return predictions, mse, mae, mse ** 0.5
