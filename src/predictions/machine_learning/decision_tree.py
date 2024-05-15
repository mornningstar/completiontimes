from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, plot_tree

from src.predictions.base_model import BaseModel


class DecisionTreeModel(BaseModel):
    def __init__(self, max_depth: int = None):
        super().__init__(DecisionTreeRegressor(max_depth=max_depth))

    def train(self, x_train, y_train=None):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, y_test, x_test=None):
        predictions = self.predict(x_test)
        return predictions, mean_squared_error(y_test, predictions)