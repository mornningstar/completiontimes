import pprint
import time

from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor

from src.predictions.base_model import BaseModel


class MLPModel(BaseModel):
    def __init__(self, auto_tune=False):
        super().__init__()
        self.auto_tune_flag = auto_tune

    @staticmethod
    def get_param_grid():
        return {
            'hidden_layer_sizes': [
                (64,), (128,), (128, 64), (128, 64, 32), (256, 128, 64)
            ],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [500],
            'early_stopping': [True]
        }

    def auto_tune(self, x_train, y_train):
        self.logger.info("Starting hyperparameter tuning...")
        start_time = time.time()

        param_grid = self.get_param_grid()

        self.logger.info("Tuning with parameter grid:")
        self.logger.info(pprint.pformat(param_grid))

        search = RandomizedSearchCV(
            estimator=MLPRegressor(random_state=42),
            param_distributions=param_grid,
            n_iter=20,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1,
            random_state=42,
            return_train_score=True
        )
        search.fit(x_train, y_train)

        elapsed_time = time.time() - start_time

        self.model = search.best_estimator_

        cv_results = search.cv_results_
        self.logger.info("Train scores:")
        self.logger.info(pprint.pformat(cv_results['mean_train_score']))
        self.logger.info("Validation scores:")
        self.logger.info(pprint.pformat(cv_results['mean_test_score']))

        self.logger.info(f"Best score: {search.best_score_:.4f}")
        self.logger.info(f"Best parameters:\n{pprint.pformat(search.best_params_)}")
        self.logger.info(f"Tuning finished in {elapsed_time:.2f} seconds")

    def train(self, x_train, y_train):
        self.logger.info("Training model..")
        x_train = self.scaler.fit_transform(x_train)

        if self.auto_tune_flag:
            self.logger.info("Tuning hyperparameters with GridSearchCV...")
            self.auto_tune(x_train, y_train)
        else:
            self.model = MLPRegressor(hidden_layer_sizes=(128, 64),
                                      activation='relu',
                                      solver='adam',
                                      max_iter=500,
                                      early_stopping=True,
                                      random_state=42)
            self.model.fit(x_train, y_train)

    def evaluate(self, x_test, y_test):
        self.logger.info("Evaluating model...")
        x_test = self.scaler.transform(x_test)
        return self.model.predict(x_test)
