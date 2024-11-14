from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller

from src.predictions.base_model import BaseModel
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel(BaseModel):

    def __init__(self):
        super().__init__()

        self.model = None
        self.fitted_model = None
        self.order = None

    def adf_test(self, y_train):
        """
        Run the augmented Dickey-Fuller Test to test for stationary. With this info, we tune the d parameter
        :param y_train:
        :return:
        """
        result = adfuller(y_train)
        p_value = result[1]
        return p_value <= 0.05  # If p-value <= 0.05, data is stationary

    def auto_tune(self, y_train):
        stationary = self.adf_test(y_train)
        if not stationary:
            d = 1  # Apply differencing if not stationary
        else:
            d = 0  # No differencing needed

        self.model = pm.auto_arima(y_train,
                                   start_p=0, max_p=5,
                                   start_q=0, max_q=4,
                                   d=d,
                                   seasonal=False,
                                   stepwise=True)
        print(self.model.summary())
        self.order = self.model.order

    def train(self, x_train, y_train):
        if not self.model:
            self.model = ARIMA(y_train, order=(1, 1, 1))
        else:
            # Use auto-tune parameters
            self.model = ARIMA(y_train, order=self.model.order)

        self.model = self.model.fit()

    def predict(self, steps):
        return self.model.forecast(steps=steps)

    def evaluate(self, x_test, y_test):
        steps = len(y_test)
        predictions = self.predict(steps)

        mse = mean_squared_error(y_true=y_test, y_pred=predictions)
        mae = mean_absolute_error(y_true=y_test, y_pred=predictions)

        return predictions, mse, mae, mse ** 0.5
