from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm

from src.predictions.base_model import BaseModel
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel(BaseModel):

    def __init__(self):
        super().__init__()

        self.model = None
        self.fitted_model = None
        self.order = None

    def auto_tune(self, x_train):
        self.model = pm.auto_arima(x_train, start_p=0, start_q=0, max_p=3, max_q=3, max_f=1, seasonal=False,
                                   stepwise=True, D=1, max_D=1, trace=True)
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
