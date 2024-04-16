import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tslearn.svm import TimeSeriesSVR
from tslearn.utils import to_time_series_dataset


class FileSizePredictor:

    def __init__(self, file_history):
        self.file_history = file_history
        self.model = TimeSeriesSVR(kernel="gak")
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.ts_dataset = None
        self.sizes_normalised = None

    def prepare_data(self):
        sizes = np.array(self.file_history['size']).reshape(-1, 1)
        self.sizes_normalised = self.scaler.fit_transform(sizes)

        self.ts_dataset = to_time_series_dataset(self.sizes_normalised)

    def train_model(self):
        if self.ts_dataset is not None and self.sizes_normalised is not None:
            X_train = self.ts_dataset[:-1]
            y_train = self.sizes_normalised[1:].ravel()  # Predicting the next size
            print("TRAINING MODEL")
            self.model.fit(X_train, y_train)
        else:
            raise ValueError("Data has not been prepared. Call prepare_data() before training.")

    def predict_next_sizes(self, num_predictions=10):
        if self.ts_dataset is not None:
            predictions = []
            future_dates = []

            last_point = self.ts_dataset[-1]
            last_date = self.file_history.index[-1]

            average_interval = (self.file_history.index[-1] - self.file_history.index[0]) / (len(self.file_history.index) - 1)

            for _ in range(num_predictions):
                last_point_reshaped = last_point.reshape(1, -1, 1)
                next_size_normalised = self.model.predict(last_point_reshaped)
                next_size = self.scaler.inverse_transform(next_size_normalised.reshape(-1, 1))[0][0]

                predictions.append(next_size)

                # Calculate next date
                next_date = last_date + average_interval
                future_dates.append(next_date)

                new_point = np.append(last_point.flatten()[1:], next_size_normalised)
                last_point = new_point

                last_date = next_date

            return predictions, future_dates
        else:
            raise ValueError("Model has not been trained. Call train_model() after prepare_data().")
