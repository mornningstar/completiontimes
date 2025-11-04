import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

from src.predictions.explainability.explainability_analyzer import ExplainabilityAnalyzer
from src.predictions.training.results.results import EvaluationMetrics, ErrorAnalysisPath, EvaluationPath


class ModelEvaluator:
    def __init__(self, model, model_plotter, output_dir, logger):
        self.model = model
        self.model_plotter = model_plotter
        self.output_dir = output_dir
        self.logger = logger

    @staticmethod
    def _calculate_smape(y_true, y_pred):
        """
        Calculates Symmetric Mean Absolute Percentage Error (sMAPE).
        """
        numerator = 2 * np.abs(y_true - y_pred)
        denominator = np.abs(y_true) + np.abs(y_pred)
        errors = numerator / denominator

        # Handle the 0/0 case: if both true and pred are 0, error is 0
        errors = np.nan_to_num(errors, nan=0.0)

        return np.mean(errors)

    def evaluate(self, x_test, y_test, test_df, feature_cols):
        y_pred_log = self.model.evaluate(x_test, y_test)

        y_pred_log = np.clip(y_pred_log, a_min=None, a_max=15)

        y_pred = np.maximum(np.expm1(y_pred_log), 0)
        
        y_test = np.expm1(y_test)

        result_df = test_df[["path", "date"]].copy()
        result_df["actual"] = y_test
        result_df["pred"] = y_pred
        eval_csv_path = os.path.join(self.output_dir, "evaluation.csv")
        result_df.to_csv(eval_csv_path, index=False)

        result_df["residual"] = result_df["actual"] - result_df["pred"]
        result_df["abs_error"] = result_df["residual"].abs()

        # ------ Calculate Metrics ------
        mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        mae_std = result_df["abs_error"].std()
        rmse = np.sqrt(mse)
        mdae = median_absolute_error(y_true=y_test, y_pred=y_pred)

        metrics = EvaluationMetrics(mse=mse, mae=mae, mae_std=mae_std, rmse=rmse, mdae=mdae)

        self.logger.info(f"Evaluation — MSE: {metrics.mse:.2f}, MAE: {metrics.mae:.2f} (std: {metrics.mae_std:.2f}), "
                         f"RMSE: {metrics.rmse:.2f}")

        extra_cols = [col for col in ["completion_reason", "committer_grouped"] if col in test_df.columns]
        feature_df = test_df[["path", "date"] + feature_cols + extra_cols]
        errors_df = pd.merge(result_df, feature_df, on=["path", "date"])

        self.model_plotter.plot_residuals(y_test, y_pred)
        self.model_plotter.plot_errors_vs_actual(y_test, y_pred)
        self.model_plotter.plot_predictions_vs_actual(y_test, y_pred)
        self.model_plotter.plot_top_errors(errors_df, n=10)

        eval_path = EvaluationPath(evaluation_path=eval_csv_path)
        return y_pred, errors_df, metrics, eval_path

    def analyze_performance_by_age(self, errors_df):
        if 'age_in_days' not in errors_df.columns:
            self.logger.warning("Skipping performance vs. age analysis: 'age_in_days' column not found.")
            return

        age_bins = [0, 30, 90, 180, 365, np.inf]
        age_labels = ["0-30", "31-90", "91-180", "181-365", "365+"]

        errors_df['age_bin'] = pd.cut(errors_df['age_in_days'], bins=age_bins, labels=age_labels, right=False)

        age_analysis = errors_df.groupby('age_bin')['abs_error'].agg(['mean', 'std']).rename(
            columns={'mean': 'mae', 'std': 'mae_std'}
        )

        self.logger.info("MAE by File Age:\n{}".format(age_analysis))

        self.model_plotter.plot_mae_by_age(age_analysis)

    @staticmethod
    def perform_error_analysis(errors_df, feature_cols, categorical_cols, model, model_plotter, output_dir, logger,
                               threshold: float = 30.0):
        errors_df["error_type"] = "ok"
        errors_df.loc[errors_df["residual"] > threshold, "error_type"] = "underestimated"
        errors_df.loc[errors_df["residual"] < -threshold, "error_type"] = "overestimated"

        errors_df["true_bin"] = pd.cut(errors_df["actual"], bins=[0, 30, 90, 180, 365, 1000],
                                       labels=["<30", "30–90", "90–180", "180–365", "365+"])

        error_counts = errors_df["error_type"].value_counts()
        logger.info("Error types:\n{}".format(error_counts))
        model_plotter.plot_error_types_pie(errors_df["error_type"])

        explain = ExplainabilityAnalyzer(model=model, feature_names=feature_cols, categorical_features=categorical_cols,
                                         model_plotter=model_plotter)
        explain.analyze_worst_predictions(errors_df, top_n=3)
        explain.analyze_best_predictions(errors_df, top_n=3)

        stats_dfs = explain.analyze_error_sources(errors_df)
        stats_by_bins, stats_by_ext, stats_by_dir, stats_by_reason, stats_by_committer = stats_dfs

        stats_by_bins.to_csv(os.path.join(output_dir, "error_stats_by_actual_bin.csv"))
        stats_by_ext.to_csv(os.path.join(output_dir, "error_stats_by_extension.csv"))
        stats_by_dir.to_csv(os.path.join(output_dir, "error_stats_by_directory.csv"))
        stats_by_reason.to_csv(os.path.join(output_dir, "error_stats_by_reason.csv"))
        if stats_by_committer is not None:
            stats_by_committer.to_csv(os.path.join(output_dir, "error_stats_by_committer.csv"))

        explain.analyze_shap_by_committer(errors_df)

        sample_for_pdp = errors_df.sample(n=min(500, len(errors_df)), random_state=42)
        x_sample = sample_for_pdp[feature_cols]

        explain.analyze_feature_interactions(x_sample)

        explain.analyze_pdp_ice(x_sample)

        error_csv = os.path.join(output_dir, "error_analysis.csv")
        errors_df.to_csv(error_csv, index=False)

        return ErrorAnalysisPath(error_analysis_path=error_csv)
