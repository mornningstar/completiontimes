import numpy as np
import keras_tuner as kt
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from keras import Sequential, Input

from src.predictions.base_model import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, timesteps=10, units=5, dropout_rate=0.5, epochs=20):
        self.num_features = None
        self.timesteps = timesteps
        self.units = units
        self.dropout_rate = dropout_rate
        self.epochs = epochs

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model = None

    def build_model(self, hp):
        units = hp.Int('units', min_value=2, max_value=100, step=16)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])
        batch_size = hp.Int('batch_size', min_value=16, max_value=64, step=16)
        kernel_regularizer = hp.Float('l2', min_value=0.0001, max_value=0.01, step=0.001)

        model = Sequential([
            Input(shape=(self.timesteps, self.num_features)),
            LSTM(units=units, return_sequences=True, kernel_regularizer=l2(kernel_regularizer)),
            Dropout(dropout_rate),
            LSTM(units=units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(units=1)
        ])

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def train(self, x_train, y_train):
        samples, timesteps, features = x_train.shape
        self.num_features = features

        x_train_reshaped = x_train.reshape(samples * timesteps, features)
        x_train_scaled = self.x_scaler.fit_transform(x_train_reshaped)
        x_train_scaled = x_train_scaled.reshape(samples, timesteps, features)

        y_train = y_train.reshape(-1, 1)
        y_train_scaled = self.y_scaler.fit_transform(y_train)
        y_train_scaled = y_train_scaled.astype(np.float32)

        tuner = kt.RandomSearch(
            self.build_model,
            objective='val_loss',
            max_trials=10,
            executions_per_trial=1)
            #directory='my_dir',
            #project_name='lstm_tuning')

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        tuner.search(x_train_scaled,
                     y_train_scaled,
                     epochs=30,
                     validation_split=0.1,
                     callbacks=[early_stop])

        self.model = tuner.get_best_models(num_models=1)[0]
        best_params_ = tuner.get_best_hyperparameters(num_trials=1)[0].values
        print(f"Best hyperparameters: {best_params_}")

        self.model.fit(x_train_scaled, y_train_scaled,
                       epochs=self.epochs,
                       batch_size=best_params_['batch_size'],
                       validation_split=0.1,
                       callbacks=[early_stop],
                       verbose=1)

    def predict(self, x_test):

        samples, timesteps, features = x_test.shape

        # Reshape and scale the test data
        x_test_reshaped = x_test.reshape(-1, features)
        x_test_scaled = self.x_scaler.transform(x_test_reshaped)
        x_test_scaled = x_test_scaled.reshape(samples, timesteps, features).astype(np.float32)

        predictions_scaled = self.model.predict(x_test_scaled)
        predictions_scaled_reshaped = predictions_scaled.reshape(-1, 1)
        predictions = self.y_scaler.inverse_transform(predictions_scaled_reshaped)

        print("predict ended")
        return predictions

    def evaluate(self, y_test, x_test):
        """
        Evaluate the model performance on test data.
        :param y_test: Ground truth labels.
        :param x_test: Test features.
        :return: Predictions, MSE, MAE, RMSE.
        """

        print("evaluate started")

        # Make predictions
        predictions = self.predict(x_test)

        # Ensure that predictions and y_test have the same shape
        min_length = min(len(predictions), len(y_test))  # Ensure we only use the matching part
        predictions = predictions[:min_length]
        y_test_trimmed = y_test[:min_length]

        # Calculate evaluation metrics
        mse = np.mean((predictions.flatten() - y_test_trimmed) ** 2)
        mae = np.mean(np.abs(predictions.flatten() - y_test_trimmed))
        rmse = np.sqrt(mse)

        print(f"Evaluate ended. MSE: {mse}, MAE: {mae}, RMSE: {rmse}")

        return predictions, mse, mae, rmse
