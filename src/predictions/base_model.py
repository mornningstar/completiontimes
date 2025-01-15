import logging

import joblib
from sklearn.preprocessing import StandardScaler


class BaseModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(self.__class__.__name__)

    def scale_data(self, data):
        return self.scaler.fit_transform(data)

    def inverse_scale(self, data):
        return self.scaler.inverse_transform(data)

    def train(self, x_train, y_train):
        raise NotImplementedError("Train method must be implemented.")

    def evaluate(self, x_test, y_test):
        raise NotImplementedError("Evaluate method must be implemented.")

    def save_model(self, filepath):
        try:
            joblib.dump(self.model, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")

    def load_model(self, filepath):
        try:
            self.model = joblib.load(filepath)
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise