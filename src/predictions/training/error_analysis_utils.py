import os

import pandas as pd

from src.predictions.explainability.explainability_analyzer import ExplainabilityAnalyzer
from src.predictions.training.results.results import ErrorAnalysisPath


def perform_error_analysis(errors_df, feature_cols, model, model_plotter, output_dir, logger, threshold: float = 30.0):
    errors_df["error_type"] = "ok"
    errors_df.loc[errors_df["residual"] > threshold, "error_type"] = "underestimated"
    errors_df.loc[errors_df["residual"] < -threshold, "error_type"] = "overestimated"

    errors_df["true_bin"] = pd.cut(errors_df["actual"], bins=[0, 30, 90, 180, 365, 1000],
                                   labels=["<30", "30–90", "90–180", "180–365", "365+"])

    error_counts = errors_df["error_type"].value_counts()
    logger.info("Error types:\n{}".format(error_counts))
    model_plotter.plot_error_types_pie(errors_df["error_type"])

    explain = ExplainabilityAnalyzer(model=model, feature_names=feature_cols, model_plotter=model_plotter)
    explain.analyze_top_errors(errors_df)
    explain.analyze_error_sources(errors_df)
    explain.analyze_shap_by_committer(errors_df)

    error_csv = os.path.join(output_dir, "error_analysis.csv")
    errors_df.to_csv(error_csv, index=False)

    return ErrorAnalysisPath(error_analysis_path=error_csv)

