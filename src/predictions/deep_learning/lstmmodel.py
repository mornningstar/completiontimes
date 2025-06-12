import logging

import numpy as np
import keras_tuner as kt
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dropout, Dense
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.python import keras

from src.predictions.base_model import BaseModel


class LSTMModel(BaseModel):

    def __init__(self):#, timesteps=10, units=5, dropout_rate=0.5, epochs=20):
        super().__init__()

        self.num_features = None
        self.model = None

    def build_model(self, hp):
        model = Sequential()
        model.add(LSTM(
            hp.Int('units', 16, 128, step=16),
            input_shape=(None, self.num_features))
        )
        model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=hp.Choice('lr', [1e-2, 1e-3, 1e-4])), loss="mse")

        return model

    def scale_data(self, y_data, fit=False):
        """
        Scaling the y data using BestModel's StandardScaler()
        :param y_data:
        :param fit: defines if the scaler needs to be fit (first time scaling) or not (refitting the model)
        :return: scaled y data
        """
        self.logger.info("Scaling data")
        if fit:
            return super().scale_data(y_data.reshape(-1, 1)).flatten()

        return self.scaler.transform(y_data.reshape(-1, 1)).flatten()


    def tune_hyperparameters(self, x_train, y_train):
        self.logger.info("Tuning hyperparameters")
        tuner = kt.RandomSearch(
            self.build_model,
            objective="val_loss",
            max_trials=5,
            overwrite=True
        )

        early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        tuner.search(
            x_train, y_train,
            epochs=10,
            validation_split=0.1,
            callbacks=[early_stop]
        )

        logging.info(f"Best hyperparameters: {tuner.get_best_hyperparameters(1)[0].values}")

        return tuner.get_best_models(1)[0]

    def train(self, x_train, y_train, refit=False):

        if x_train.ndim != 3 or y_train.ndim != 1:
            raise ValueError("x_train must have shape (samples, timesteps, features) and y_train must be 1D.")

        if len(x_train) != len(y_train):
            raise ValueError("x_train and y_train must have the same number of samples.")

        self.num_features = x_train.shape[2]

        y_train_scaled = self.scale_data(y_train, fit=not refit)

        if not refit:
            # Hyperparameter tuning and training
            self.logger.info("Performing initial training...")
            self.model = self.tune_hyperparameters(x_train, y_train_scaled)

        self.model.fit(x_train, y_train_scaled, epochs=10, validation_split=0.1, verbose=1)
        logging.info("Model training/refitting completed.")

    def predict(self, x_test):
        predictions = self.model.predict(x_test).flatten()

        return self.inverse_scale(predictions.reshape(-1, 1)).flatten()

    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)

        mse = np.mean((y_test - predictions) ** 2)
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(mse)

        return predictions, mse, mae, rmse

    def save_best_model(self, filepath):
        self.model.save(filepath)
        print(f"Best model saved at {filepath}")

    def load_best_model(self, filepath):
        self.model = keras.models.load_model(filepath)
        print(f"Best model loaded from {filepath}")