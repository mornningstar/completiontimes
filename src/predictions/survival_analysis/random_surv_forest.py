import pprint
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from sksurv.ensemble import RandomSurvivalForest

from src.predictions.base_model import BaseModel


def _format_labels(y):
    """Convert (duration, event) to structured array for sksurv."""
    duration = y[:, 0]
    event = y[:, 1].astype(bool)
    return np.array([(e, d) for e, d in zip(event, duration)], dtype=[('event', bool), ('duration', float)])


class RandomSurvivalForestModel(BaseModel):
    """
    RandomForest as a model for survival analysis. Does not allow time-varying data points.
    """
    def __init__(self, auto_tune_flag: bool = False):
        super().__init__()
        self.fitted_event_times_ = None
        self.auto_tune_flag = auto_tune_flag
        self.model = None
    
    def auto_tune(self, x_train, y_train, groups, cv=5, n_trials=100, timeout=None):

        df = pd.DataFrame({"file": groups, "event": y_train[:, 1]})
        grp_event = df.groupby("file")["event"].max().to_dict()

        skf = StratifiedGroupKFold(n_splits=cv)

        #unique_groups = np.unique(groups)
        #test_size = max(1, int(0.2 * len(unique_groups)))
        #splitter = GroupTimeSeriesSplit(test_size=test_size, n_splits=cv)

        for i, (train_idx, test_idx) in enumerate(skf.split(x_train, y=[grp_event[f] for f in groups], groups=groups)):
            y_train_fold = y_train[train_idx]
            y_test_fold = y_train[test_idx]

            p_train = np.mean(y_train_fold[:, 1])
            p_test = np.mean(y_test_fold[:, 1])

            print(f"Fold {i + 1}: event=1 in train={p_train:.2%}, in test={p_test:.2%}")

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

            score = cross_val_score(model,x_train,y_train_struct,groups=groups,
                cv=skf.split(
                    x_train,
                    y=[grp_event[f] for f in groups],
                    groups=groups
                ),
                n_jobs=-1)

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

        self.fitted_event_times_ = np.unique(y_train[:, 0])
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

        if times is None:
            if not hasattr(self, "fitted_event_times_"):
                raise AttributeError("No event times available. Ensure train() was called.")
            times = self.fitted_event_times_

        surv_funcs = self.model.predict_survival_function(x)
        return [(times, sf(times)) for sf in surv_funcs]

    def predict_risk(self, x, horizon=None):
        curves = self.predict(x)
        if horizon is not None:
            return np.array([1 - np.interp(horizon, t, s) for t, s in curves])
        else:
            return np.array([-np.trapz(s, t) for t, s in curves])