from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationMetrics:
    """ Container for evaluation metrics. """
    mse: float
    mae: float
    mae_std: float
    rmse: float
    mdae: float

@dataclass
class TrainingResult:
    """ Container for evaluation results. """
    model: Any
    metrics: EvaluationMetrics
    model_path: str
    evaluation_csv: str
    error_analysis_csv: str

@dataclass
class ErrorAnalysisPath:
    """ Container for error analysis paths. """
    error_analysis_path: str

@dataclass
class EvaluationPath:
    """ Container for evaluation paths. """
    evaluation_path: str