import logging
import pprint
import time

import numpy as np
import optuna
from mlxtend.evaluate import GroupTimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from src.predictions.base_model import BaseModel


class GradientBoosting(BaseModel):
    def __init__(self, auto_tune=False):
        super().__init__()

        self.model = None
        self.auto_tune_flag = auto_tune

        self.logger = logging.getLogger(self.__class__.__name__)

    def auto_tune(self, x_train, y_train, groups, cv=5, scoring='neg_mean_squared_error', n_trials = 150, timeout = None):
        self.logger.info("GradientBoosting - Starting hyperparameter tuning...")

        unique_groups = np.unique(groups)
        num_groups = len(unique_groups)

        test_size = max(1, int(num_groups * 0.2))

        splitter = GroupTimeSeriesSplit(test_size=test_size, n_splits=cv)

        def objective(trial):
            param_grid = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            }

            model = GradientBoostingRegressor(random_state=42, **param_grid)
            scores = cross_val_score(model, x_train, y_train, groups=groups, cv=splitter,
                                     scoring=scoring, n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction='maximize' if scoring.startswith("neg_") else 'minimize')
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        elapsed_time = time.time() - start_time

        self.model = GradientBoostingRegressor(random_state=42, **study.best_params)
        self.model.fit(x_train, y_train)

        self.logger.info(f"Best score: {study.best_value:.4f} ({scoring})")
        self.logger.info("Best parameters:\n" + pprint.pformat(study.best_params))
        self.logger.info(f"Tuning finished in {elapsed_time:.2f} seconds")

        return study.best_params

    def train(self, x_train, y_train, groups = None):
        self.logger.info("GradientBoosting - Training model..")

        if self.auto_tune_flag:
            if groups is None:
                raise ValueError("Groups are required for auto_tuning.")
            self.logger.info("Tuning hyperparameters with Optuna...")
            self.auto_tune(x_train, y_train, groups=groups)
        else:
            self.model = GradientBoostingRegressor(random_state=42)
            self.model.fit(x_train, y_train)
            self.logger.info("Training completed.")

    def evaluate(self, x_test, y_test):
        self.logger.info("GradientBoosting - Evaluating model..")
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(x_test)