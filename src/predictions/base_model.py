from sklearn.metrics import mean_squared_error


class BaseModel:
    def __init__(self, model=None):
        self.model = model

    def train(self, X_train, Y_train=None):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, X_test=None):
        predictions =self.predict(X_test)
        return mean_squared_error(y_true=y_test, y_pred=predictions)