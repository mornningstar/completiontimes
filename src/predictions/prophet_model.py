import logging

import numpy as np
import optuna
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

from src.predictions.base_model import BaseModel


class ProphetModel(BaseModel):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.seasonal = None
        self.best_params = None

    def tune_hyperparameters(self, df, n_trials=50):
        """
        Uses Optuna to tune hyperparameters for Prophet model.

        Parameters:
            df (DataFrame): Data containing 'ds' and 'y' columns for Prophet.
            n_trials (int): Number of trials for Optuna study.

        Returns:
            dict: Best hyperparameters found by Optuna.
        """

        def objective(trial):
            params = {
                "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
                "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
            }
            model = Prophet(**params)
            model.fit(df)

            df_cv = cross_validation(model, initial='365 days', horizon='30 days')
            return performance_metrics(df_cv)["mse"].mean()

        # Create an Optuna study and optimize
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_params
        logging.info("Found best parameters for Prohet model with MSE: {}".format(study.best_value))

        return self.best_params

    def train(self, x_train, y_train):
        # Prophet requires a DataFrame with specific column names
        df_train = pd.DataFrame({"ds": x_train, "y": y_train}).dropna()

        if not self.best_params:
            self.best_params = self.tune_hyperparameters(df_train)

        self.model = Prophet(**self.best_params)
        self.model.fit(df_train)

    def predict(self, x_test):
        # Prophet expects a DataFrame with 'ds' column for dates
        df_future = pd.DataFrame({"ds": x_test})
        return self.model.predict(df_future)["yhat"].values

    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)

        mse = mean_squared_error(y_true=y_test, y_pred=predictions)
        mae = mean_absolute_error(y_true=y_test, y_pred=predictions)
        rmse = np.sqrt(mse)

        return predictions, mse, mae, rmse
