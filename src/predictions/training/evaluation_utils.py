import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.predictions.training.results.results import EvaluationMetrics, EvaluationPath


def evaluate_regression_predictions(x_test, y_test, test_df, model, model_plotter, output_dir, logger, feature_cols):
    y_pred_log = model.evaluate(x_test, y_test)

    max_days = 1_000
    max_safe_log = np.log1p(max_days)  # ≃ 6.91
    y_pred_log = np.clip(y_pred_log, a_min=None, a_max=max_safe_log)
    y_pred = np.maximum(np.expm1(y_pred_log), 0)
    y_test = np.expm1(y_test)

    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    rmse = np.sqrt(mse)
    metrics = EvaluationMetrics(mse=mse, mae=mae, rmse=rmse)

    logger.info(f"Evaluation — MSE: {metrics.mse:.2f}, MAE: {metrics.mae:.2f}, RMSE: {metrics.rmse:.2f}")

    result_df = test_df[["path", "date"]].copy()
    result_df["actual"] = y_test
    result_df["pred"] = y_pred
    eval_csv_path = os.path.join(output_dir, "evaluation.csv")
    result_df.to_csv(eval_csv_path, index=False)

    result_df["residual"] = result_df["actual"] - result_df["pred"]
    result_df["abs_error"] = result_df["residual"].abs()

    extra_cols = [col for col in ["completion_reason", "committer_grouped"] if col in test_df.columns]
    feature_df = test_df[["path", "date"] + feature_cols + extra_cols]
    errors_df = pd.merge(result_df, feature_df, on=["path", "date"])

    model_plotter.plot_residuals(y_test, y_pred)
    model_plotter.plot_errors_vs_actual(y_test, y_pred)
    model_plotter.plot_predictions_vs_actual(y_test, y_pred)
    model_plotter.plot_top_errors(y_test, y_pred, n=10)

    eval_path = EvaluationPath(evaluation_path=eval_csv_path)
    return y_pred, errors_df, metrics, eval_path