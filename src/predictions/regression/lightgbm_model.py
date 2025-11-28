import pprint
import time

import optuna
from lightgbm import LGBMRegressor
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score

from src.predictions.base_model import BaseModel
from src.predictions.model_tuner import ModelTuner


class LightGBMModel(BaseModel):
    def __init__(self, auto_tune=True):
        super().__init__()
        self.model = None
        self.auto_tune_flag = auto_tune

        self.model_tuner = ModelTuner()

    def auto_tune(self, x_train, y_train, groups, n_trials = 100, cv = 5, scoring='neg_mean_absolute_error',
                  timeout=None, split_strategy='by_file'):
        self.logger.info(f"Starting LightGBM hyperparameter tuning with '{split_strategy}' strategy...")

        splitter, cv_groups = self.model_tuner.get_splitter(split_strategy, groups, cv)

        def objective(trial):
            trial_params = {
                'objective': 'regression_l1',
                'metric': 'mae',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 256),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'n_jobs': 1
            }

            model = LGBMRegressor(**trial_params, verbose=-1)
            scores = cross_val_score(model, x_train, y_train, groups=cv_groups, cv=splitter,
                            scoring=scoring, n_jobs=(self.CPU_LIMIT // 8))
            return scores.mean()

        best_params = self.model_tuner.tune(objective, scoring, n_trials, timeout)
        final_params = {**best_params, 'random_state': 42, 'n_jobs': self.CPU_LIMIT // 8}
        self.model = LGBMRegressor(**final_params)


    def train(self, x_train, y_train, groups = None, split_strategy='by_file'):
        self.logger.info("LightGBM: Training model..")
        if self.auto_tune_flag:
            if groups is None and split_strategy == 'by_file':
                raise ValueError("Groups are required for auto_tuning.")

            self.logger.info("Tuning hyperparameters with Optuna...")
            self.auto_tune(x_train, y_train, groups, split_strategy=split_strategy)
        else:
            self.model = LGBMRegressor(random_state=42)

        self.model.fit(x_train, y_train)

    def evaluate(self, x_test, y_test):
        return self.model.predict(x_test)