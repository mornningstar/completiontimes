from sklearn.metrics import mean_absolute_error, mean_squared_error


class BaseModel:
    def __init__(self, model=None):
        self.model = model

    def train(self, x_train, y_train):
        if y_train is not None:
            self.model.fit(x_train, y_train)
        else:
            self.model.fit(x_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, y_test, x_test):
        predictions = self.predict(x_test)

        mse = mean_squared_error(y_true=y_test, y_pred=predictions)
        mae = mean_absolute_error(y_true=y_test, y_pred=predictions)

        return predictions, mse, mae, mse ** 0.5
