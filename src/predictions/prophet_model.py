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

        self.seasonal = None
        self.best_params = None
        self.model = None

    def tune_hyperparameters(self, df, n_trials=100):
        """
        Uses Optuna to tune hyperparameters for Prophet model.

        Parameters:
            df (DataFrame): Data containing 'ds' and 'y' columns for Prophet.
            n_trials (int): Number of trials for Optuna study.

        Returns:
            dict: Best hyperparameters found by Optuna.
        """

        def objective(trial):
            # Suggest values for each hyperparameter
            seasonality_mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
            changepoint_prior_scale = trial.suggest_loguniform("changepoint_prior_scale", 0.001, 0.5)
            seasonality_prior_scale = trial.suggest_loguniform("seasonality_prior_scale", 0.1, 20)
            holidays_prior_scale = trial.suggest_loguniform("holidays_prior_scale", 0.1, 20)
            yearly_seasonality = trial.suggest_categorical("yearly_seasonality", [5, 10, 15, False])
            weekly_seasonality = trial.suggest_categorical("weekly_seasonality", [3, 7, 10, False])
            daily_seasonality = trial.suggest_categorical("daily_seasonality", [3, 5, 7, False])

            # Initialize Prophet model with the current set of hyperparameters
            model = Prophet(
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality
            )

            # Fit the model
            model.fit(df)

            # Perform cross-validation
            df_cv = cross_validation(model, initial='365 days', period='180 days', horizon='30 days')
            df_performance = performance_metrics(df_cv)

            # Use mean MSE as the evaluation metric
            mse = df_performance["mse"].mean()
            return mse

        # Create an Optuna study and optimize
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_params
        print("Best hyperparameters:", self.best_params)
        print("Best MSE:", study.best_value)
        return self.best_params

    def train(self, x_train, y_train):
        # Prophet requires a DataFrame with specific column names
        df_train = pd.DataFrame({'ds': x_train, 'y': y_train})
        df_train.dropna(subset=['ds', 'y'], inplace=True)

        if not self.best_params:
            self.best_params = self.tune_hyperparameters(df_train)

        #self.model = Prophet()
        # Initialize Prophet with best parameters
        self.model = Prophet(
            seasonality_mode=self.best_params["seasonality_mode"],
            changepoint_prior_scale=self.best_params["changepoint_prior_scale"],
            seasonality_prior_scale=self.best_params["seasonality_prior_scale"],
            holidays_prior_scale=self.best_params["holidays_prior_scale"],
            yearly_seasonality=self.best_params["yearly_seasonality"],
            weekly_seasonality=self.best_params["weekly_seasonality"],
            daily_seasonality=self.best_params["daily_seasonality"]
        )

        self.model.fit(df_train)

    def predict(self, x_test):
        # Prophet expects a DataFrame with 'ds' column for dates
        future = pd.DataFrame({'ds': x_test})
        forecast = self.model.predict(future)

        # Prophet outputs a DataFrame with a column 'yhat' for predictions
        predictions = forecast['yhat'].values
        return predictions

    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)

        mse = mean_squared_error(y_true=y_test, y_pred=predictions)
        mae = mean_absolute_error(y_true=y_test, y_pred=predictions)
        rmse = np.sqrt(mse)

        return predictions, mse, mae, rmse
