import numpy as np

from src.predictions.base_model import BaseModel


class MedianBaseModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.median = None

    def train(self, x_train, y_train, **kwargs):
        self.median = np.median(y_train)
        self.logger.info(f"Trained MedianBaselineModel with median = {self.median:.2f}")
    
    def evaluate(self, x_test, y_test):
        return np.full_like(y_test, self.median)

    def predict(self, x_test):
        return np.full(len(x_test), self.median)