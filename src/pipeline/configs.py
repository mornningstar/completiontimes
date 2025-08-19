from src.data_handling.features.regression_feature_eng import RegressionFeatureEngineering
from src.data_handling.features.survival_feature_engineer import SurvivalFeatureEngineer
from src.predictions.training.regression_model_trainer import RegressionModelTrainer
from src.predictions.training.survival_model_trainer import SurvivalModelTrainer

ENGINEER_BY_TYPE = {
    "regression": RegressionFeatureEngineering,
    "survival":   SurvivalFeatureEngineer,
}

TRAINER_BY_TYPE = {
    "regression": RegressionModelTrainer,
    "survival":   SurvivalModelTrainer,
}
