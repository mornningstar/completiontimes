import logging
import os

import numpy as np

from src.visualisations.model_plotting import ModelPlotter


class FileModelTrainer:
    def __init__(self, project_name, model, output_dir="models"):
        self.project_name = project_name
        self.model = model(auto_tune=True)

        self.output_dir = output_dir

        self.model_plotter = ModelPlotter(project_name)

        self.logger = logging.getLogger(self.__class__.__name__)

        os.makedirs(output_dir, exist_ok=True)


    def split_by_file(self, file_data_df, test_ratio=0.2, random_state=42):
        file_data_df = file_data_df.copy()
        valid_paths = file_data_df[file_data_df["days_until_completion"].notna()]["path"].unique()

        # Shuffle paths to avoid any ordering bias
        np.random.seed(random_state)
        np.random.shuffle(valid_paths)

        split_idx = int(len(valid_paths) * (1 - test_ratio))
        train_paths = valid_paths[:split_idx]
        test_paths = valid_paths[split_idx:]

        train_df = file_data_df[file_data_df["path"].isin(train_paths)].dropna(subset=["days_until_completion"])
        test_df = file_data_df[file_data_df["path"].isin(test_paths)].dropna(subset=["days_until_completion"])

        return train_df, test_df

    def get_feature_cols(self, file_data_df, include_size=False):
        drop_cols = ["path", "date", "completion_date", "completion_reason", "days_until_completion"]
        if not include_size:
            drop_cols.append("size")

        return [col for col in file_data_df.select_dtypes(include=np.number).columns if col not in drop_cols]

    def train_and_evaluate(self, file_data_df):
        train_df, test_df = self.split_by_file(file_data_df)

        feature_cols = self.get_feature_cols(train_df)
        self.logger.info(f"Used feature columns: {feature_cols}")

        x_train = train_df[feature_cols].values
        y_train = np.log1p(train_df["days_until_completion"].values) #train_df["days_until_completion"].values
        x_test = test_df[feature_cols].values
        y_test = np.log1p(test_df["days_until_completion"].values) #test_df["days_until_completion"].values

        #x_train_scaled = self.model.scale_data(x_train)
        #x_test_scaled = self.model.scaler.transform(x_test)

        #self.model.train(x_train_scaled, y_train)
        self.model.train(x_train, y_train)

        #y_pred, mse, mae, rmse = self.model.evaluate(x_test_scaled, y_test)
        y_pred, mse, mae, rmse, mape = self.model.evaluate(x_test, y_test)
        self.logger.info(f"MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}")

        self.model_plotter.plot_residuals(y_test, y_pred)

        model_path = os.path.join(self.output_dir, f"{self.model.__class__.__name__}.pkl")
        self.model.save_model(model_path)

        return {
            "model": self.model,
            "mae": mae,
            "rmse": rmse,
            "model_path": model_path
        }
