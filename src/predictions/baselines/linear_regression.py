from sklearn.linear_model import LinearRegression

from src.predictions.base_model import BaseModel


class LinearRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LinearRegression()

    def train(self, x_train, y_train, **kwargs):
        x_scaled = self.scale_data(x_train)
        self.model.fit(x_scaled, y_train)
        self.logger.info("Linear Regression model trained.")

    def evaluate(self, x_test, y_test, **kwargs):
        x_scaled = self.scaler.transform(x_test)
        predictions = self.model.predict(x_scaled)

        return predictions
