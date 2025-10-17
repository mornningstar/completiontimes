from src.predictions.baselines.linear_regression import LinearRegressionModel
from src.predictions.baselines.median_base_model import MedianBaseModel
from src.predictions.regression.lightgbm_model import LightGBMModel
from src.predictions.regression.gradient_boosting import GradientBoosting
from src.predictions.regression.random_forest import RandomForestModel
from src.predictions.regression.xgboost import XGBoostModel

MODEL_REGISTRY = {
    "LinearRegressionModel": LinearRegressionModel,
    "MedianBaseModel": MedianBaseModel,
    "LightGBMModel": LightGBMModel,
    "RandomForestModel": RandomForestModel,
    "GradientBoosting": GradientBoosting,
    "XGBoostModel": XGBoostModel,
}

def get_model_class(class_name: str):
    """
    Retrieves a model class from the registry
    """
    if class_name not in MODEL_REGISTRY:
        raise ValueError(f"Model class '{class_name}' not found in the registry.")

    return MODEL_REGISTRY[class_name]
