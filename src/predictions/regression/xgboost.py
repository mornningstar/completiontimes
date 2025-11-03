import pprint
import time

import numpy as np
import optuna
import xgboost
from mlxtend.evaluate import GroupTimeSeriesSplit
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from config import globals

from src.predictions.base_model import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self, auto_tune=False):
        super().__init__()
        self.auto_tune_flag = auto_tune

    def auto_tune(self, x_train, y_train, groups, cv=5, scoring='neg_mean_squared_error', n_trials=100, timeout=None,
                  split_strategy='by_file'):
        self.logger.info(f"Starting hyperparameter tuning with '{split_strategy}' strategy...")

        if split_strategy == 'by_file':
            unique_groups = np.unique(groups)
            num_groups = len(unique_groups)
            test_size = max(1, int(num_groups * 0.2))
            splitter = GroupTimeSeriesSplit(test_size=test_size, n_splits=cv)
            cv_groups = groups
        elif split_strategy == "by_history":
            splitter = TimeSeriesSplit(n_splits=cv)
            cv_groups = None
        else:
            raise ValueError(f"Unknown split_strategy: {split_strategy}")

        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42
            }
            model = xgboost.XGBRegressor(**params)
            score = cross_val_score(model, x_train, y_train, groups=cv_groups, cv=splitter, scoring=scoring,
                                    n_jobs=globals.CPU_LIMIT // 8)
            return score.mean()

        study = optuna.create_study(direction='maximize')
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=8)
        elapsed_time = time.time() - start_time

        self.model = xgboost.XGBRegressor(random_state=42, **study.best_params)
        self.model.fit(x_train, y_train)

        self.logger.info(f"Best score: {study.best_value:.4f} ({scoring})")
        self.logger.info("Best parameters:\n" + pprint.pformat(study.best_params))
        self.logger.info(f"Tuning finished in {elapsed_time:.2f} seconds")

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
            self.logger.info("XGBoost: Training completed.")

    def evaluate(self, x_test, y_test, **kwargs):
        self.logger.info("XGBoost: Evaluating model...")
        if self.model is None:
            raise ValueError("XGBoost: Model is not trained yet.")
        return self.model.predict(x_test)