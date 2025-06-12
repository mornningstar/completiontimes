import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.predictions.explainability.explainability_analyzer import ExplainabilityAnalyzer
from src.visualisations.model_plotting import ModelPlotter


class RegressionModelTrainer:
    def __init__(self, project_name, model, images_dir, output_dir="models"):
        self.project_name = project_name
        self.model = model(auto_tune=True)

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model_plotter = ModelPlotter(project_name, model, images_dir=images_dir)

        self.logger = logging.getLogger(self.__class__.__name__)

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
        drop_cols = ["path", "date", "completion_date", "completion_reason", "days_until_completion", "committer",
                     "committer_grouped"]
        if not include_size:
            drop_cols.append("size")

        return [col for col in file_data_df.select_dtypes(include=np.number).columns if col not in drop_cols]

    def train(self, x_train, y_train, groups=None):
        self.model.train(x_train, y_train, groups=groups)

    def evaluate(self, x_test, y_test, test_df):
        y_pred_log = self.model.evaluate(x_test, y_test)

        max_days = 1_000
        max_safe_log = np.log1p(max_days)  # ≃ 6.91
        y_pred_log = np.clip(y_pred_log, a_min=None, a_max=max_safe_log)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(y_pred, 0)
        y_test = np.expm1(y_test)

        mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        rmse = np.sqrt(mse)

        self.logger.info(f"Evaluation — MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        result_df = test_df[["path", "date"]].copy()
        result_df["actual"] = y_test
        result_df["pred"] = y_pred
        result_df.to_csv(f"{self.output_dir}/evaluation.csv", index=False)

        result_df["residual"] = result_df["actual"] - result_df["pred"]
        result_df["abs_error"] = result_df["residual"].abs()
        feature_cols = self.get_feature_cols(test_df)

        extra_cols = [col for col in ["completion_reason", "committer_grouped"] if col in test_df.columns]
        feature_df = test_df[["path", "date"] + feature_cols + extra_cols]

        errors_df = pd.merge(result_df, feature_df, on=["path", "date"])
        errors_df.to_csv(f"{self.output_dir}/error_analysis.csv", index=False)
        self.logger.info(f"Saved error analysis at {self.output_dir}/error_analysis.csv")

        self.model_plotter.plot_residuals(y_test, y_pred)
        self.model_plotter.plot_errors_vs_actual(y_test, y_pred)

        self.model_plotter.plot_predictions_vs_actual(y_test, y_pred)
        self.model_plotter.plot_top_errors(y_test, y_pred, n=10)

        return y_pred, mse, mae, rmse, errors_df

    def train_and_evaluate(self, file_data_df):
        train_df, test_df = self.split_by_file(file_data_df)
        feature_cols = self.get_feature_cols(train_df)
        self.logger.info(f"Used feature columns: {feature_cols}")

        x_train = train_df[feature_cols]
        x_test = test_df[feature_cols]

        y_train_log = np.log1p(train_df["days_until_completion"].values)
        y_test_log = np.log1p(test_df["days_until_completion"].values)
        
        self.logger.debug(f"Length of x_train: {len(x_train)}")
        self.logger.debug(f"Length of y_train: {len(y_train_log)}")

        groups = train_df["path"].values
        self.train(x_train, y_train_log, groups=groups)

        importances = self.model.get_feature_importances()
        if importances is not None:
            self.model_plotter.plot_model_feature_importance(feature_cols, importances)

        #Evaluation
        y_pred, mse, mae, rmse, errors_df = self.evaluate(x_test, y_test_log, test_df)

        self.run_error_analysis(errors_df, feature_cols)

        model_path = os.path.join(self.output_dir, f"{self.model.__class__.__name__}.pkl")
        self.model.save_model(model_path)

        results_csv_path = os.path.join(self.output_dir, "results.csv")
        results_data = {
            "project": self.project_name,
            "model": self.model.__class__.__name__,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "model_path": model_path
        }
        
        if os.path.exists(results_csv_path):
            existing_df = pd.read_csv(results_csv_path)
            results_df = pd.concat([existing_df, pd.DataFrame([results_data])], ignore_index=True)
        else:
            results_df = pd.DataFrame([results_data])

        results_df.to_csv(results_csv_path, index=False)
        

        return {
            "model": self.model,
            "mae": mae,
            "rmse": rmse,
            "model_path": model_path
        }

    def predict_unlabeled_files(self, file_data_df, latest_only=False):
        unlabeled_df = file_data_df[file_data_df["days_until_completion"].isna()].copy()

        if latest_only:
            unlabeled_df = unlabeled_df.sort_values('date').groupby('path').tail(1).reset_index(drop=True)

        if unlabeled_df.empty:
            self.logger.info("No unlabeled files to predict.")
            return pd.DataFrame()

        feature_cols = self.get_feature_cols(unlabeled_df)
        self.logger.info(f"Predicting on {len(unlabeled_df)} unlabeled rows using features: {feature_cols}")

        x_unlabeled = unlabeled_df[feature_cols]

        predictions = self.model.predict(x_unlabeled)
        predictions = np.clip(predictions, a_min=None, a_max=15) # expm1(15) ≈ 3.2 million days
        predictions = np.expm1(predictions)

        unlabeled_df["predicted_days_until_completion"] = predictions
        unlabeled_df["completion_date_predicted"] = unlabeled_df["date"] + pd.to_timedelta(predictions, unit="D")

        result = unlabeled_df[["path", "date", "predicted_days_until_completion"]]

        result.to_csv(f"{self.output_dir}/prediction.csv", index=False)
        self.logger.info(f"Predictions saved to {self.output_dir}")

        return result

    def run_error_analysis(self, errors_df, feature_cols, threshold=30.0):
        errors_df["error_type"] = "ok"
        errors_df.loc[errors_df["residual"] > threshold, "error_type"] = "underestimated"
        errors_df.loc[errors_df["residual"] < -threshold, "error_type"] = "overestimated"

        errors_df["true_bin"] = pd.cut(errors_df["actual"], bins=[0, 30, 90, 180, 365, 1000],
                                       labels=["<30", "30–90", "90–180", "180–365", "365+"])


        error_counts = errors_df["error_type"].value_counts()
        self.logger.info("Error types:\n{}".format(error_counts))
        self.model_plotter.plot_error_types_pie(errors_df["error_type"])

        explain = ExplainabilityAnalyzer(model=self.model, feature_names=feature_cols, model_plotter=self.model_plotter)
        explain.analyze_top_errors(errors_df)
        explain.analyze_error_sources(errors_df)
        explain.analyze_shap_by_committer(errors_df)