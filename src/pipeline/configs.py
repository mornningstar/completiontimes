from src.data_handling.features.regression_feature_eng import RegressionFeatureEngineering
from src.predictions.training.regression_model_trainer import RegressionModelTrainer

ENGINEER_BY_TYPE = {
    "regression": RegressionFeatureEngineering,
}

TRAINER_BY_TYPE = {
    "regression": RegressionModelTrainer,
}
