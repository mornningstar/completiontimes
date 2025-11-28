import pprint
import time

import numpy as np
import optuna
from mlxtend.evaluate import GroupTimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from src.predictions.base_model import BaseModel
from src.predictions.model_tuner import ModelTuner


class RandomForestModel(BaseModel):
    def __init__(self, auto_tune=False):
        super().__init__()
        self.auto_tune_flag = auto_tune

        self.model_tuner = ModelTuner()

    def auto_tune(self, x_train, y_train, groups, cv=5, scoring='neg_mean_absolute_error', n_trials=100,
                  timeout=None, split_strategy='by_file'):
        self.logger.info(f"Starting hyperparameter tuning with '{split_strategy}' strategy...")

        splitter, cv_groups = self.model_tuner.get_splitter(split_strategy, groups, cv)

        def objective(trial):
            params = {
                'criterion': 'squared_error',
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 25, step=5),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.75]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.2),
                'n_jobs': self.CPU_LIMIT,
            }

            if params['bootstrap']:
                params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)
            else:
                params['max_samples'] = None

            model = RandomForestRegressor(random_state=42, **params)
            score = cross_val_score(model, x_train, y_train, groups=cv_groups, cv=splitter, scoring=scoring)
            return score.mean()

        best_params = self.model_tuner.tune(objective, scoring, n_trials, timeout)
        self.model = RandomForestRegressor(random_state=42, **best_params)

    def train(self, x_train, y_train, groups=None, split_strategy='by_file'):
        self.logger.info("RandomForest: Training model..")

        if self.auto_tune_flag:
            if groups is None and split_strategy == 'by_file':
                raise ValueError("Groups are required for auto_tuning.")
            self.logger.info("Tuning hyperparameters with Optuna...")
            self.auto_tune(x_train, y_train, groups=groups, split_strategy=split_strategy)
        else:
            self.model = RandomForestRegressor(random_state=42)
            self.logger.info("Training completed.")

        self.model.fit(x_train, y_train)

    def evaluate(self, x_test, y_test, **kwargs):
        self.logger.info("Evaluating model...")
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(x_test)
