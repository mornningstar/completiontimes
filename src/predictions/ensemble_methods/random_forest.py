import pprint
import time

import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.predictions.base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, auto_tune=False):
        super().__init__()
        self.auto_tune_flag = auto_tune

    def auto_tune(self, x_train, y_train, cv=5, scoring='neg_mean_squared_error', n_trials=150,
                  timeout=None):
        self.logger.info("Starting hyperparameter tuning...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30, step=5),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.05),
            }

            if params['bootstrap']:
                params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)
            else:
                params['max_samples'] = None

            model = RandomForestRegressor(random_state=42, **params)
            score = cross_val_score(model, x_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            return score.mean()
    
        study = optuna.create_study(direction='maximize' if 'neg' in scoring else 'minimize')
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        elapsed_time = time.time() - start_time

        self.model = RandomForestRegressor(random_state=42, **study.best_params)
        self.model.fit(x_train, y_train)

        self.logger.info(f"Best score: {study.best_value:.4f} ({scoring})")
        self.logger.info("Best parameters:\n" + pprint.pformat(study.best_params))
        self.logger.info(f"Tuning finished in {elapsed_time:.2f} seconds")

        return study.best_params

    def train(self, x_train, y_train):
        self.logger.info("Training model..")

        if self.auto_tune_flag:
            self.logger.info("Tuning hyperparameters with Optuna...")
            self.auto_tune(x_train, y_train)
        else:
            self.model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
            self.model.fit(x_train, y_train)
            self.logger.info("Training completed.")

    def evaluate(self, x_test, y_test):
        self.logger.info("Evaluating model...")
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(x_test)

