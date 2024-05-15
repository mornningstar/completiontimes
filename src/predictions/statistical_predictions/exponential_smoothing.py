from sklearn.metrics import mean_squared_error

from src.predictions.base_model import BaseModel
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


class SimpleExponentialSmoothing(BaseModel):
    def __init__(self):
        super().__init__()

    def train(self, x_train, y_train):
        self.model = SimpleExpSmoothing(y_train, initialization_method='estimated').fit()

    def predict(self, steps):
        return self.model.forecast(steps=steps)

    def evaluate(self, y_test, x_test):
        predictions = self.predict(len(x_test))
        return predictions, mean_squared_error(y_test, predictions)



