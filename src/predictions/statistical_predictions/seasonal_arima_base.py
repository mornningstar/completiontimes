import numpy as np
import pandas as pd
import pmdarima as pmd
from scipy.signal import periodogram
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
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
        self.logger.info(f"ADF Test p-value: {p_value}")

        return p_value <= 0.05  # If p-value <= 0.05, data is stationary

    def detect_seasonality(self, data, max_lag=365, min_seasonality=7):
        data = np.asarray(data, dtype=np.float64)

        acf_values = acf(data, nlags=max_lag, fft=True)
        acf_values[0] = 0
        acf_values[:min_seasonality + 1] = 0
        seasonal_lag_acf = np.argmax(acf_values[:max_lag])

        frequencies, spectrum = periodogram(data)
        seasonal_lag_fft = int(1 / frequencies[np.argmax(spectrum)]) if len(frequencies) > 1 else None

        # Combine results
        if acf_values[seasonal_lag_acf] > 0.5:
            self.logger.info(f"ACF detected seasonality: {seasonal_lag_acf} days")
            return seasonal_lag_acf
        elif seasonal_lag_fft and seasonal_lag_fft <= max_lag:
            self.logger.info(f"FFT detected seasonality: {seasonal_lag_fft} days")
            return seasonal_lag_fft
        else:
            self.logger.info("No significant seasonality detected")
            return None

    def scale_data(self, data):
        """Scale data using StandardScaler."""
        if isinstance(data, pd.Series):  # Convert to NumPy array if it's a pandas Series
            data = data.values

        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        return data_scaled

    def inverse_scale(self, data):
        """Inverse the scaling."""
        data_original = self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        return data_original

    def auto_tune(self, y_train):
        """
        Auto-tune the model by detecting seasonality and setting ARIMA or SARIMA parameters.
        """

        # Check if the data is stationary
        stationary = self.adf_test(y_train)
        d = 0 if stationary else 1

        # Detect seasonality
        seasonal_period = self.detect_seasonality(y_train)
        sufficient_samples = (len(y_train) > (seasonal_period + 10)) if seasonal_period else False

        if seasonal_period and sufficient_samples:
            self.logger.info("Seasonality detected. Trying seasonal ARIMA.")

            try:
                self.model = pmd.auto_arima(
                    y_train, start_p=0, start_q=0, max_p=3, max_q=3,
                    d=d, seasonal=True, m=seasonal_period,
                    stepwise=True, trace=True
                )
            except ValueError as e:
                if "There are no more samples" in str(e):
                    self.logger.warning(
                        "Seasonal differencing failed due to insufficient samples; "
                        "falling back to non-seasonal ARIMA."
                    )

                    self.model = pmd.auto_arima(
                        y_train, start_p=0, max_p=5, start_q=0, max_q=4, d=d,
                        seasonal=False, stepwise=True
                    )

                else:
                    raise e
        else:
            self.logger.info(
                "Either no seasonality detected or insufficient data for seasonal differencing. "
                "Using non-seasonal ARIMA."
            )

            self.model = pmd.auto_arima(
                y_train, start_p=0, max_p=5, start_q=0, max_q=4,
                d=d, seasonal=False, stepwise=True
            )

        self.order = self.model.order
        self.seasonal_order = getattr(self.model, 'seasonal_order', None)

        self.logger.info(f"Selected ARIMA order: {self.order}")
        if self.seasonal_order:
            self.logger.info(f"Selected SARIMA seasonal order: {self.seasonal_order}")

    def train(self, x_train, y_train, refit=False):
        """
        Train the model using either ARIMA or SARIMA.
        """
        self.logger.info(f"Training model with order={self.order} and seasonal_order={self.seasonal_order}")

        y_train_scaled = self.scale_data(y_train)
        self.auto_tune(y_train_scaled)

        if self.seasonal_order:
            # Use SARIMA
            self.fitted_model = ARIMA(y_train_scaled, order=self.order, seasonal_order=self.seasonal_order).fit()
        else:
            # Use ARIMA
            self.fitted_model = ARIMA(y_train_scaled, order=self.order).fit()

    def predict(self, future):
        """
        Generate predictions based on the fitted model.
        """
        if self.fitted_model is None:
            raise ValueError("Model is not trained. Call 'train' before predicting.")

        if not isinstance(future, int):
            steps = len(future)
        else:
            steps = future

        scaled_predictions = self.fitted_model.forecast(steps=steps)
        return self.inverse_scale(scaled_predictions)

    def evaluate(self, x_test, y_test):
        """
        Evaluate the model's performance.
        """
        predictions = self.predict(len(y_test))

        mse = mean_squared_error(y_true=y_test, y_pred=predictions)
        mae = mean_absolute_error(y_true=y_test, y_pred=predictions)
        rmse = mse ** 0.5

        self.logger.info(f"Evaluation - MSE: {mse}, MAE: {mae}, RMSE: {rmse}")

        return predictions, mse, mae, rmse