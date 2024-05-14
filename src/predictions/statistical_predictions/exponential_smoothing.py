from sklearn.metrics import mean_squared_error

from src.predictions.base_model import BaseModel
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


class SimpleExponentialSmoothing(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, x_train, y_train=None):
        self.model = SimpleExpSmoothing(x_train, initialization_method='estimated').fit()

    def predict(self, steps):
        return self.model.forecast(steps=steps)

    def evaluate(self, y_test, x_test=None):
        predictions = self.predict(len(y_test))
        return predictions, mean_squared_error(y_test, predictions)



