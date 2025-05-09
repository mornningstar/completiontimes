import pprint
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from src.predictions.base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, auto_tune=False):
        super().__init__()
        self.auto_tune_flag = auto_tune

    def auto_tune(self, x_train, y_train, param_grid=None, cv=5, scoring='neg_mean_squared_error', n_jobs=-1):
        self.logger.info("Starting hyperparameter tuning...")

        if param_grid is None:
            param_grid = {
                #{
                    # if bootstrap true -> max_samples is allowed, otherwise not
                    'bootstrap': [True],
                    'max_samples': [None, 0.7, 0.85],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 25, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', 0.3, 0.5],
                #},
                #{
                #    # If bootstrap false -> max_samples must be None
                #    'bootstrap': [False],
                #    'max_samples': [None],
                #    'n_estimators': [100, 200, 300],
                #    'max_depth': [10, 15, 25, None],
                #    'min_samples_split': [2, 5, 10],
                #    'min_samples_leaf': [1, 2, 4],
                #    'max_features': ['sqrt', 'log2', 0.3, 0.5],
                #}
            }

        self.logger.info("Tuning with parameter grid:")
        self.logger.info(pprint.pformat(param_grid))

        start_time = time.time()

        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs
        )

        grid_search.fit(x_train, y_train)

        elapsed_time = time.time() - start_time

        # Update the model with the best parameters
        self.model = grid_search.best_estimator_

        self.logger.info(f"Best score: {grid_search.best_score_:.4f} ({scoring})")
        self.logger.info(f"Best parameters:\n{pprint.pformat(grid_search.best_params_)}")
        self.logger.info(f"Tuning finished in {elapsed_time:.2f} seconds")

        return grid_search.best_params_

    def train(self, x_train, y_train):
        self.logger.info("Training model..")

        if self.auto_tune_flag:
            self.logger.info("Tuning hyperparameters with GridSearchCV...")
            self.auto_tune(x_train, y_train)
        else:
            self.model = RandomForestRegressor(n_estimators=100, max_depth=None,
                                               random_state=42)
            self.model.fit(x_train, y_train)
            self.logger.info("Training completed.")

    def evaluate(self, x_test, y_test):
        self.logger.info("Evaluating model...")
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(x_test)

