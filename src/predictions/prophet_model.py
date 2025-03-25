import logging

import numpy as np
import optuna
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.predictions.base_model import BaseModel


class ProphetModel(BaseModel):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.seasonal = None
        self.best_params = None

    def tune_hyperparameters(self, df, n_trials=100, patience=10):
        best_trial = None
        best_mse = float("inf")
        no_improvement_count = 0

        def objective(trial):
            nonlocal best_trial, best_mse, no_improvement_count

            params = {
                "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
                "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 1, log=True),
                "weekly_seasonality": trial.suggest_categorical("weekly_seasonality", [True, False]),
                "yearly_seasonality": trial.suggest_categorical("yearly_seasonality", [True, False]),
            }

            model = Prophet(**params)

            model.fit(df)

            train_proportion = 0.7  # 70% for initial window
            data_span = (df['ds'].max() - df['ds'].min()).days
            initial_window = f"{int(data_span * train_proportion)} days"

            calculated_horizon = int(data_span * 0.2)
            horizon_days = max(7, min(calculated_horizon, max(30, int(data_span * 0.1))))

            df_cv = cross_validation(model, initial=initial_window, horizon=f"{horizon_days} days")

            mse = performance_metrics(df_cv)["mse"].mean()

            # **Early stopping logic**
            if mse < best_mse:
                best_mse = mse
                best_trial = trial
                no_improvement_count = 0
                logging.info(f"New best MSE: {best_mse} found at trial {trial.number}")
            else:
                no_improvement_count += 1
                logging.info(f"No improvement (count: {no_improvement_count}), MSE: {mse}")

            if no_improvement_count >= patience:
                logging.info(f"Early stopping triggered at trial {trial.number}, best MSE: {best_mse}")
                raise optuna.exceptions.TrialPruned()  # Stop further trials

            return mse

        # Create an Optuna study and optimize
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials, n_jobs=4)

        self.best_params = study.best_params
        logging.info("Found best parameters for Prophet model with MSE: {}".format(study.best_value))

        return self.best_params

    def train(self, x_train, y_train, refit=False):
        if len(x_train) != len(y_train):
            raise ValueError(f"Mismatch: x_train({len(x_train)}) != y_train({len(y_train)})")

        df_train = pd.DataFrame({"ds": x_train, "y": y_train}).dropna()

        df_train["ds"] = df_train["ds"].dt.tz_localize(None)

        if not self.best_params and not refit:
            self.best_params = self.tune_hyperparameters(df_train)

        self.logger.info("Training Prophet model...")

        self.model = Prophet(**self.best_params)
        self.model.fit(df_train)

        self.logger.info("Model training/refitting completed.")

    def predict(self, x_test):
        df_future = pd.DataFrame({"ds": x_test})
        df_future["ds"] = pd.to_datetime(df_future["ds"]).dt.tz_localize(None)

        predictions = self.model.predict(df_future)["yhat"].values

        self.logger.info(f"Generated predictions for {len(x_test)} data points.")

        return predictions

    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)
        mse = mean_squared_error(y_true=y_test, y_pred=predictions)
        mae = mean_absolute_error(y_true=y_test, y_pred=predictions)
        rmse = np.sqrt(mse)

        self.logger.info(f"Evaluation completed - MSE: {mse}, MAE: {mae}, RMSE: {rmse}")

        return predictions, mse, mae, rmse
