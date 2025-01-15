import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from src.predictions.base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators, max_depth, random_state=42):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)

        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self.model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                            random_state=random_state)

    def auto_tune(self, x_train, y_train, param_grid=None, cv=5, scoring='neg_mean_squared_error', n_jobs=-1):
        self.logger.info("Starting hyperparameter tuning...")

        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }

        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=self.default_params['random_state']),
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs
        )

        scaled_x_train = self.scale_data(x_train)
        grid_search.fit(scaled_x_train, y_train)

        # Update the model with the best parameters
        self.model = grid_search.best_estimator_
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best score: {grid_search.best_score_}")

        return grid_search.best_params_

    def train(self, x_train, y_train):
        self.logger.info("Training RandomForest model")
        scaled_x_train = self.scale_data(x_train)
        self.model.fit(scaled_x_train, y_train)
        self.logger.info("Training completed.")

    def evaluate(self, x_test, y_test):
        self.logger.info("Evaluating RandomForest model")
        scaled_x_test = self.scale_data(x_test)
        predictions = self.model.predict(scaled_x_test)

        mse = mean_squared_error(y_true=y_test, y_pred=predictions)
        mae = mean_absolute_error(y_true=y_test, y_pred=predictions)
        rmse = np.sqrt(mse)

        return predictions, mse, mae, rmse

