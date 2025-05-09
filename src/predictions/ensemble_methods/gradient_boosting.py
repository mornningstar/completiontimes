import logging
import pprint
import time

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from src.predictions.base_model import BaseModel


class GradientBoosting(BaseModel):
    def __init__(self, auto_tune=False):
        super().__init__()

        self.model = None
        self.auto_tune_flag = auto_tune

        self.logger = logging.getLogger(self.__class__.__name__)

    def auto_tune(self, x_train, y_train):
        self.logger.info("GradientBoosting - Starting hyperparameter tuning...")

        scoring = "neg_mean_squared_error"

        param_grid = {
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [200, 300, 400],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.5]
        }

        self.logger.info("Tuning with parameter grid:")
        self.logger.info(pprint.pformat(param_grid))

        start_time = time.time()

        grid_search = GridSearchCV(
            estimator=GradientBoostingRegressor(random_state=42),
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1
        )

        grid_search.fit(x_train, y_train)
        elapsed_time = time.time() - start_time

        self.model = grid_search.best_estimator_

        self.logger.info(f"Best score: {grid_search.best_score_:.4f} ({scoring})")
        self.logger.info(f"Best parameters:\n{pprint.pformat(grid_search.best_params_)}")
        self.logger.info(f"Tuning finished in {elapsed_time:.2f} seconds")

        return grid_search.best_params_

    def train(self, x_train, y_train):
        self.logger.info("GradientBoosting - Training model..")

        if self.auto_tune_flag:
            self.logger.info("Tuning hyperparameters with GridSearchCV...")
            self.auto_tune(x_train, y_train)
        else:
            self.model.fit(x_train, y_train)
            self.logger.info("Training completed.")

    def evaluate(self, x_test, y_test):
        self.logger.info("GradientBoosting - Evaluating model..")

        return self.model.predict(x_test)