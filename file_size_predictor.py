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
        #sizes = np.array([record['size'] for record in self.file_history]).reshape(-1, 1)
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

    def predict_next_size(self):
        if self.ts_dataset is not None:
            last_point = self.ts_dataset[-1].reshape(1, -1, 1)
            print("PREDICTING NEXT POINT")
            next_size_normalised = self.model.predict(last_point)
            next_size = self.scaler.inverse_transform(next_size_normalised.reshape(-1, 1))[0][0]

            last_date = self.file_history.index[-1]
            average_interval = (self.file_history.index[-1] - self.file_history.index[0]) / (len(self.file_history.index) - 1)
            next_date = last_date + average_interval

            return next_size, next_date
        else:
            raise ValueError("Model has not been trained. Call train_model() after prepare_data().")
