import logging
import traceback

import pandas as pd

from src.predictions.machine_learning.lstmmodel import LSTMModel


class ModelTrainer:
    def __init__(self, model_classes, modeling_tasks=None, save_models=True):
        self.model_classes = model_classes
        self.modeling_tasks = modeling_tasks
        self.save_models = save_models

        self.logger = logging.getLogger(__name__)

    def train_and_evaluate_model(self, x_train, y_train, x_test, y_test, use_clusters=False, refit_full=False):

        model_info = {}

        for model_class in self.model_classes:
            model_name = model_class.__class__.__name__

            try:

                self.logger.info(f"Starting training for {model_name}")
                model = model_class()

                # Hyperparameter Tuning
                if hasattr(model, 'auto_tune'):
                    self.logger.info(f"Tuning parameters for {model_name}")
                    model.auto_tune(y_train)
                if hasattr(model, 'param_grid'):
                    self.logger.info(f"Grid searching for {model_name}")
                    params = model.grid_search(x_train, y_train)
                    self.logger.info(f"Best params for {model_name}: {params}")

                # Train the model
                if isinstance(model, LSTMModel):
                    model.train(x_train, y_train)
                    model.save_best_model(f"{model.__class__.__name__}_trained_model.h5")
                else:
                    model.train(x_train, y_train)

                self.logger.info(f"{model_name} training completed.")

                # Evaluate the model
                predictions, mse, mae, rmse = model.evaluate(x_test, y_test)
                self.logger.info(f"{model_name} evaluation - MSE: {mse}, MAE: {mae}, RMSE: {rmse}")

                if self.save_models and hasattr(model, 'save_model'):
                    model.save_model(f"{model_name}_trained_model.pkl")
                    self.logger.info(f"Saved {model_name} model to disk.")

                model_info[model.__class__.__name__] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'predictions': predictions,
                    'x_train': x_train,
                    'y_train': y_train,
                    'x_test': x_test,
                    'y_test': y_test
                }

                self.logger.info(f"Training for {model_name} completed with MSE: {mse}.")

            except Exception as e:
                self.logger.error(f"Error during training for {model_name}: {str(e)}")
                self.logger.error(traceback.format_exc())

        return model_info

    def refit_model(self, model, x, y, steps=30):
        self.logger.info(f"Refitting {model.__class__.__name__} on full data...")

        last_date = x[-1] if isinstance(x, pd.DatetimeIndex) else x.index[-1]

        # LSTM-specific reshaping
        if isinstance(model, LSTMModel):
            x = x.values.reshape(-1, x.shape[1], x.shape[2])
            y = y.values

        model.train(x_train=x, y_train=y, refit=True)

        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq="D")[1:]

        predictions = model.predict(future_dates)
        self.logger.info(f"{model.__class__.__name__} refitting and prediction completed.")

        return future_dates, predictions