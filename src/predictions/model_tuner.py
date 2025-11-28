import logging
import pprint
import time

import numpy as np
import optuna
from mlxtend.evaluate import GroupTimeSeriesSplit
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit


class ModelTuner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_splitter(self, split_strategy, groups, cv):
        if split_strategy == 'by_file':
            unique_groups = np.unique(groups)
            num_groups = len(unique_groups)
            test_size = max(1, int(num_groups * 0.2))
            splitter = GroupTimeSeriesSplit(test_size=test_size, n_splits=cv)
            cv_groups = groups
        elif split_strategy == 'by_history':
            splitter = TimeSeriesSplit(n_splits=cv)
            cv_groups = None
        else:
            raise ValueError(f"Unknown split_strategy: {split_strategy}")

        return splitter, cv_groups

    def tune(self, objective_func, scoring, n_trials, timeout):
        study = optuna.create_study(direction='maximize' if scoring.startswith("neg_") else 'minimize',
                                    sampler=TPESampler(seed=42))
        start_time = time.time()
        study.optimize(objective_func, n_trials=n_trials, timeout=timeout, n_jobs=8)
        elapsed_time = time.time() - start_time

        self.logger.info(f"Best score: {study.best_value:.4f} ({scoring})")
        self.logger.info("Best parameters:\n" + pprint.pformat(study.best_params))
        self.logger.info(f"Tuning finished in {elapsed_time:.2f} seconds")

        return study.best_params