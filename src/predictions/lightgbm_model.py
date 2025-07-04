import pprint
import time

import lightgbm
import numpy as np
import optuna
from lightgbm import LGBMRegressor
from mlxtend.evaluate import GroupTimeSeriesSplit
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error

from src.predictions.base_model import BaseModel


class LightGBMModel(BaseModel):
    def __init__(self, auto_tune=True):
        super().__init__()
        self.model = None
        self.auto_tune_flag = auto_tune

    def auto_tune(self, x_train, y_train, groups, n_trials = 75, cv = 5):
        self.logger.info("Starting hyperparameter tuning...")

        unique_groups = np.unique(groups)
        num_groups = len(unique_groups)
        test_size = max(1, int(num_groups * 0.2))

        splitter = GroupTimeSeriesSplit(test_size=test_size, n_splits=cv)

        def objective(trial):
            trial_params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 512),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),

                'objective': trial.suggest_categorical('objective', ['quantile', 'regression', 'regression_l1']),
                'random_state': 42
            }

            if trial_params['objective'] == 'quantile':
                trial_params['alpha'] = 0.5

            maes = []

            for train_idx, valid_idx in splitter.split(x_train, y_train, groups=groups):
                X_train, X_val = x_train.iloc[train_idx], x_train.iloc[valid_idx]
                Y_train, Y_val = y_train[train_idx], y_train[valid_idx]
                m = LGBMRegressor(**trial_params, verbose=-1)
                m.fit(X_train, Y_train,
                      eval_set=[(X_val, Y_val)],
                      eval_metric="mae",
                      callbacks=[lightgbm.early_stopping(stopping_rounds=75)])
                preds = m.predict(X_val)
                maes.append(mean_absolute_error(Y_val, preds))
            return np.mean(maes)

        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials)
        elapsed_time = time.time() - start_time

        best_objective = study.best_params.get('objective', 'regression')

        final_params = {**study.best_params, 'random_state': 42}
        if final_params['objective'] == 'quantile':
            final_params['alpha'] = 0.5

        self.model = LGBMRegressor(**final_params)
        self.logger.info(f"Best score: {study.best_value:.4f}")
        self.logger.info("Best parameters:\n" + pprint.pformat(study.best_params))
        self.logger.info(f"Tuning finished in {elapsed_time:.2f} seconds")

        return study.best_params

    def train(self, x_train, y_train, groups = None):
        if self.auto_tune_flag:
            self.logger.info("Tuning hyperparameters with Optuna...")
            self.auto_tune(x_train, y_train, groups)

        self.model.fit(x_train, y_train)

    def evaluate(self, x_test, y_test):
        return self.model.predict(x_test)