import pmdarima as pm

from src.predictions.statistical_predictions.arima import ARIMAModel


class SARIMAModel(ARIMAModel):
    def __init__(self):
        super().__init__()

    def auto_tune(self, X_train):
        """
        Auto-tune SARIMA model with seasonal=True
        :param X_train:
        :return:
        """
        self.model = pm.auto_arima(X_train, start_p=0, start_q=0, max_p=3, max_q=3, max_f=1, seasonal=True,
                                   stepwise=True, D=1, max_D=1, trace=True)
        print(self.model.summary())
        self.order = self.model.order