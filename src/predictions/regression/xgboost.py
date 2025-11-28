import pprint
import time

import numpy as np
import optuna
import xgboost
from mlxtend.evaluate import GroupTimeSeriesSplit
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from src.predictions.base_model import BaseModel
from src.predictions.model_tuner import ModelTuner


class XGBoostModel(BaseModel):
    def __init__(self, auto_tune=False):
        super().__init__()
        self.auto_tune_flag = auto_tune

        self.model_tuner = ModelTuner()

    def auto_tune(self, x_train, y_train, groups, cv=5, scoring='neg_mean_absolute_error', n_trials=100, timeout=None,
                  split_strategy='by_file'):
        self.logger.info(f"Starting hyperparameter tuning with '{split_strategy}' strategy...")

        splitter, cv_groups = self.model_tuner.get_splitter(split_strategy, groups, cv)

        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42,
                'n_jobs': 1
            }
            model = xgboost.XGBRegressor(**params)
            score = cross_val_score(model, x_train, y_train, groups=cv_groups, cv=splitter, scoring=scoring,
                                    n_jobs=1)
            return score.mean()

        best_params = self.model_tuner.tune(objective, scoring, n_trials, timeout)
        self.model = xgboost.XGBRegressor(random_state=42, **best_params)

    def train(self, x_train, y_train, groups=None, split_strategy='by_file'):
        self.logger.info("XGBoost: Training model..")

        if self.auto_tune_flag:
            if groups is None and split_strategy == 'by_file':
                raise ValueError("Groups are required for auto_tuning with XGBoost.")
            self.logger.info("XGBoost: Tuning hyperparameters with Optuna...")
            self.auto_tune(x_train, y_train, groups=groups, split_strategy=split_strategy)
        else:
            self.model = xgboost.XGBRegressor(random_state=42)

        self.model.fit(x_train, y_train)

    def evaluate(self, x_test, y_test, **kwargs):
        self.logger.info("XGBoost: Evaluating model...")
        if self.model is None:
            raise ValueError("XGBoost: Model is not trained yet.")
        return self.model.predict(x_test)