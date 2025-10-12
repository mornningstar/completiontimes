import logging
import os

import numpy as np
import pandas as pd

from src.predictions.training.data_splitter import DataSplitter
from src.predictions.training.model_evaluator import ModelEvaluator
from src.predictions.training.results.results import TrainingResult
from src.visualisations.model_plotting import ModelPlotter


class RegressionModelTrainer:
    def __init__(self, project_name, model_cfg, images_dir, output_dir="models"):
        self.project_name = project_name
        model_cls = model_cfg["class"]
        self.model = model_cls(auto_tune=True)
        self.split_strategy = model_cfg.get("split_strategy", "by_file")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model_plotter = ModelPlotter(project_name, images_dir=images_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.evaluator = ModelEvaluator(self.model, self.model_plotter, self.output_dir, self.logger)

    @staticmethod
    def get_feature_cols(file_data_df, include_size=False):
        drop_cols = ["path", "date", "completion_date", "completion_reason", "days_until_completion", "committer"]
        if not include_size:
            drop_cols.append("size")

        return [col for col in file_data_df.select_dtypes(include=np.number).columns if col not in drop_cols]


    def train_and_evaluate(self, data_tuple):
        file_data_df, categorical_cols = data_tuple

        if self.split_strategy == "by_file":
            train_df, test_df = DataSplitter.split_by_file(file_data_df)
        elif self.split_strategy == "by_history":
            train_df, test_df = DataSplitter.split_by_history(file_data_df)
        else:
            raise ValueError(f"Unknown split_strategy: {self.split_strategy}")

        # Calculate and save dataset statistics
        train_files = train_df['path'].nunique()
        test_files = test_df['path'].nunique()
        train_commits = len(train_df)
        test_commits = len(test_df)
        total_files = file_data_df['path'].nunique()

        stats = {
            'train_files': [train_files],
            'test_files': [test_files],
            'total_files': [total_files],
            'train_commits': [train_commits],
            'test_commits': [test_commits],
        }

        stats_df = pd.DataFrame(stats)
        stats_csv_path = os.path.join(self.output_dir, "dataset_statistics.csv")
        stats_df.to_csv(stats_csv_path, index=False)
        self.logger.info(f"Dataset statistics saved to {stats_csv_path}")

        all_feature_cols = self.get_feature_cols(train_df)
        numerical_cols = [col for col in all_feature_cols if col not in categorical_cols]

        self.logger.info(f"Identified {len(numerical_cols)} numerical features to be scaled.")
        self.logger.info(f"Identified {len(categorical_cols)} categorical/binary features to pass through.")

        x_train_numerical = train_df[numerical_cols]
        x_train_categorical = train_df[categorical_cols]
        x_test_numerical = test_df[numerical_cols]
        x_test_categorical = test_df[categorical_cols]

        self.model.scaler.fit(x_train_numerical)
        x_train_numerical_scaled = pd.DataFrame(self.model.scaler.transform(x_train_numerical),
                                                index=x_train_numerical.index, columns=numerical_cols)
        x_test_numerical_scaled = pd.DataFrame(self.model.scaler.transform(x_test_numerical),
                                               index=x_test_numerical.index, columns=numerical_cols)

        x_train = pd.concat([x_train_numerical_scaled, x_train_categorical], axis=1)
        x_test = pd.concat([x_test_numerical_scaled, x_test_categorical], axis=1)

        self.logger.info(f"Length of x_train: {len(x_train.columns)}; Length of x_test: {len(x_test.columns)}")

        # Ensure column order is the same
        x_test = x_test[x_train.columns]
        final_feature_cols = x_train.columns.tolist()

        if not train_df.empty and "days_until_completion" in train_df.columns:
            adaptive_cap = np.percentile(train_df["days_until_completion"].dropna(), 99)
            adaptive_cap = max(adaptive_cap, 365)
        else:
            adaptive_cap = 2000

        self.logger.info(
            f"Using an adaptive prediction cap of {adaptive_cap:.0f} days (based on 99th percentile of training data)."
        )

        y_train_log = np.log1p(train_df["days_until_completion"].values)
        y_test_log = np.log1p(test_df["days_until_completion"].values)

        groups = train_df["path"].values
        self.model.train(x_train, y_train_log, groups=groups, split_strategy=self.split_strategy)

        if self.model.model is not None:
            self.model_plotter.plot_learning_curves(self.model.model, x_train, y_train_log, groups=groups)

            importances = self.model.get_feature_importances()
            if importances is not None:
                self.model_plotter.plot_model_feature_importance(final_feature_cols, importances)

        #Evaluation
        y_pred, errors_df, metrics, eval_path = self.evaluator.evaluate(x_test, y_test_log, test_df,
                                                                        final_feature_cols, max_days=adaptive_cap)
        error_path = self.evaluator.perform_error_analysis(errors_df, final_feature_cols, self.model,
                                                           self.model_plotter, self.output_dir, self.logger)

        self.logger.info(f"Performing analysis of performance vs. file age for '{self.split_strategy}' split...")
        self.evaluator.analyze_performance_by_age(errors_df)

        model_path = os.path.join(self.output_dir, f"{self.model.__class__.__name__}.pkl")
        self.model.save_model(model_path)

        results_csv_path = os.path.join(self.output_dir, "results.csv")
        results_data = {
            "project": self.project_name,
            "model": self.model.__class__.__name__,
            "split_strategy": self.split_strategy,
            "mse": round(metrics.mse, 4),
            "mae": round(metrics.mae, 4),
            "mae_std": round(metrics.mae_std, 4),
            "rmse": round(metrics.rmse, 4),
            "model_path": model_path,
        }
        
        if os.path.exists(results_csv_path):
            existing_df = pd.read_csv(results_csv_path)
            results_df = pd.concat([existing_df, pd.DataFrame([results_data])], ignore_index=True)
        else:
            results_df = pd.DataFrame([results_data])

        results_df.to_csv(results_csv_path, index=False)

        return TrainingResult(
            model=self.model,
            metrics=metrics,
            model_path=model_path,
            evaluation_csv=eval_path,
            error_analysis_csv=error_path,
        )

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

        predictions_log = self.model.predict(x_unlabeled)

        max_days = 1_000
        max_safe_log = np.log1p(max_days)  # ≃ 6.91
        predictions_log = np.clip(predictions_log, a_min=None, a_max=max_safe_log)

        predictions = np.expm1(predictions_log)
        predictions = np.maximum(predictions, 0)

        unlabeled_df["predicted_days_until_completion"] = predictions
        unlabeled_df["completion_date_predicted"] = unlabeled_df["date"] + pd.to_timedelta(predictions, unit="D")

        result = unlabeled_df[["path", "date", "predicted_days_until_completion"]]

        result.to_csv(f"{self.output_dir}/prediction.csv", index=False)
        self.logger.info(f"Predictions saved to {self.output_dir}")

        return result
