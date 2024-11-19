import numpy as np
import pmdarima as pmd
from scipy.signal import periodogram
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf

from src.predictions.base_model import BaseModel


class SeasonalARIMABase(BaseModel):
    def __init__(self):
        super().__init__()

        self.fitted_model = None
        self.order = None
        self.seasonal_order = None

    def adf_test(self, y_train):
        """
        Run the augmented Dickey-Fuller Test to test for stationary. With this info, we tune the d parameter
        :param y_train:
        :return:
        """
        result = adfuller(y_train)
        p_value = result[1]

        return p_value <= 0.05  # If p-value <= 0.05, data is stationary

    def detect_seasonality(self, data, max_lag=365, min_seasonality=7):
        acf_values = acf(data, nlags=max_lag, fft=True)
        acf_values[0] = 0  # Ignore lag-0
        seasonal_lag_acf = np.argmax(acf_values[:max_lag])

        # Ignore lags below the minimum seasonality threshold
        acf_values[:min_seasonality] = 0

        frequencies, spectrum = periodogram(data)
        seasonal_lag_fft = int(1 / frequencies[np.argmax(spectrum)]) if len(frequencies) > 1 else None

        # Combine results
        if acf_values[seasonal_lag_acf] > 0.5:
            print(f"ACF detected seasonality: {seasonal_lag_acf} days")
            return seasonal_lag_acf
        elif seasonal_lag_fft and seasonal_lag_fft <= max_lag:
            print(f"FFT detected seasonality: {seasonal_lag_fft} days")
            return seasonal_lag_fft
        else:
            print("No significant seasonality detected")
            return None

    def auto_tune(self, y_train):
        """
        Auto-tune the model by detecting seasonality and setting ARIMA or SARIMA parameters.
        """
        # Check if the data is stationary
        stationary = self.adf_test(y_train)
        d = 0 if stationary else 1

        # Detect seasonality
        seasonal_period = self.detect_seasonality(y_train, max_lag=365)

        if seasonal_period > 52:  # Cap seasonal period to 52 weeks
            print(f"Seasonal period {seasonal_period} too large. Reducing to 52.")
            seasonal_period = 52

        # Choose ARIMA or SARIMA based on seasonality detection
        try:
            if seasonal_period and seasonal_period <= 365:
                print(f"Using SARIMA model with seasonality period: {seasonal_period}")
                self.model = pmd.auto_arima(
                    y_train, start_p=0, start_q=0, max_p=3, max_q=3,
                    d=d, seasonal=True, m=seasonal_period,
                    start_P=0, start_Q=0, max_P=2, max_Q=2,
                    stepwise=True, max_d=2, D=1, max_D=1, trace=True
                )
                self.order = self.model.order
                self.seasonal_order = self.model.seasonal_order
            else:
                print("Using non-seasonal ARIMA model.")
                self.model = pmd.auto_arima(
                    y_train, start_p=0, max_p=5,
                    start_q=0, max_q=4, d=d,
                    seasonal=False, stepwise=True
                )
                self.order = self.model.order

            print(self.model.summary())
        except MemoryError as e:
            print(f"Memory error: {e}.")

    def train(self, x_train=None, y_train=None):
        """
        Train the model using either ARIMA or SARIMA.
        """
        if y_train is None:
            raise ValueError("y_train cannot be None for ARIMA/SARIMA models.")

        if self.model is None:
            self.auto_tune(y_train)

        if self.order and self.seasonal_order:
            # Use SARIMA
            self.fitted_model = ARIMA(y_train, order=self.order, seasonal_order=self.seasonal_order).fit()
        else:
            # Use ARIMA
            self.fitted_model = ARIMA(y_train, order=self.order).fit()

    def predict(self, steps):
        """
        Generate predictions based on the fitted model.
        """
        if self.fitted_model is None:
            raise ValueError("Model is not trained. Call 'train' before predicting.")

        return self.fitted_model.forecast(steps=steps)

    def evaluate(self, x_test, y_test):
        """
        Evaluate the model's performance.
        """
        steps = len(y_test)
        predictions = self.predict(steps)

        mse = mean_squared_error(y_true=y_test, y_pred=predictions)
        mae = mean_absolute_error(y_true=y_test, y_pred=predictions)
        rmse = mse ** 0.5

        return predictions, mse, mae, rmse