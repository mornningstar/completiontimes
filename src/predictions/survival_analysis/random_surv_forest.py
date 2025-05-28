import pprint
import time

import numpy as np
import optuna
from mlxtend.evaluate import GroupTimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sksurv.ensemble import RandomSurvivalForest

from src.predictions.base_model import BaseModel


def _format_labels(y):
    """Convert (duration, event) to structured array for sksurv."""
    duration = y[:, 0]
    event = y[:, 1].astype(bool)
    return np.array([(e, d) for e, d in zip(event, duration)], dtype=[('event', bool), ('duration', float)])


class RandomSurvivalForestModel(BaseModel):
    def __init__(self, auto_tune_flag: bool = False):
        super().__init__()
        self.auto_tune_flag = auto_tune_flag
        self.model = None
    
    def auto_tune(self, x_train, y_train, groups, cv=5, n_trials=100, timeout=None):
        splitter = GroupTimeSeriesSplit(n_splits=cv)
        y_train_struct = _format_labels(y_train)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 30, step=5),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5])
            }

            model = RandomSurvivalForest(random_state=42, n_jobs=-1, **params)

            score = cross_val_score(
                model, x_train, y_train_struct, groups=groups,
                cv=splitter, scoring="concordance_index", n_jobs=-1
            )

            return score.mean()

        study = optuna.create_study(direction='maximize')
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        elapsed_time = time.time() - start_time

        best_params = study.best_params
        self.model = RandomSurvivalForest(random_state=42, n_jobs=-1, **best_params)
        self.model.fit(x_train, y_train_struct)

        self.logger.info(f"Best score: {study.best_value:.4f} (concordance_index)")
        self.logger.info("Best parameters:\n" + pprint.pformat(best_params))
        self.logger.info(f"Tuning finished in {elapsed_time:.2f} seconds")
        return best_params

    def train(self, x_train, y_train, groups=None):
        self.logger.info("Training model..")

        if self.auto_tune_flag:
            self.logger.info("Tuning hyperparameters with Optuna...")
            self.auto_tune(x_train, y_train, groups)
        else:
            y_train_structured = _format_labels(y_train)
            self.model = RandomSurvivalForest(n_estimators=100, random_state=42, n_jobs=-1)
            self.model.fit(x_train, y_train_structured)
            self.logger.info("Training completed.")


    def evaluate(self, x_test, y_test):
        self.logger.info("Evaluating model...")
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        y_test_structured = _format_labels(y_test)
        return self.model.score(x_test, y_test_structured)

    def predict(self, x, times=None):
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        surv_funcs = self.model.predict_survival_function(x)
        if times is None:
            times = self.model.event_times_
        return [(times, sf(times)) for sf in surv_funcs]

    def predict_risk(self, x, horizon=None):
        curves = self.predict(x)
        if horizon is not None:
            return np.array([1 - np.interp(horizon, t, s) for t, s in curves])
        else:
            return np.array([-np.trapz(s, t) for t, s in curves])