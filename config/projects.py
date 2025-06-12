from src.predictions.baselines.linear_regression import LinearRegressionModel
from src.predictions.baselines.median_base_model import MedianBaseModel
from src.predictions.regression.gradient_boosting import GradientBoosting
from src.predictions.lightgbm_model import LightGBMModel
from src.predictions.regression.random_forest import RandomForestModel

PROJECTS = [
    {
        'name': 'flairNLP/fundus',
        'source_directory': 'src',
        'get_newest': False,
        'models': [
            {
                "class": LinearRegressionModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": MedianBaseModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": LightGBMModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": RandomForestModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": GradientBoosting,
                "use_categorical": True,
                "feature_type": "regression"
            }
        ]
    },
    {
        'name': 'khoj-ai/khoj',
        'source_directory': 'src',
        'get_newest': False,
        'models': [
            {
                "class": LinearRegressionModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": MedianBaseModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": LightGBMModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": RandomForestModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": GradientBoosting,
                "use_categorical": True,
                "feature_type": "regression"
            }
        ]
    },
    {
        'name': 'vuejs/core',
        'source_directory': 'packages',
        'get_newest': False,
        'models': [
            {
                "class": LinearRegressionModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": MedianBaseModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": LightGBMModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": RandomForestModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": GradientBoosting,
                "use_categorical": True,
                "feature_type": "regression"
            }
        ]
    },
    {
        'name': 'mozilla/addons-server',
        'source_directory': 'src',
        'get_newest': False,
        'models': [
            {
                "class": LinearRegressionModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": MedianBaseModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": LightGBMModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": RandomForestModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": GradientBoosting,
                "use_categorical": True,
                "feature_type": "regression"
            }
        ]
    },
    {
        'name': 'fastapi/fastapi',
        'source_directory': 'fastapi',
        'get_newest': False,
        'models': [
            {
                "class": LinearRegressionModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": MedianBaseModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": LightGBMModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": RandomForestModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": GradientBoosting,
                "use_categorical": True,
                "feature_type": "regression"
            }
        ]
    },
    {
        'name': 'pallets/flask',
        'source_directory': 'src',
        'get_newest': False,
        'models': [
            {
                "class": LinearRegressionModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": MedianBaseModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": LightGBMModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": RandomForestModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": GradientBoosting,
                "use_categorical": True,
                "feature_type": "regression"
            }
        ]
    },
    {
        'name': 'keras-team/keras',
        'source_directory': 'keras',
        'get_newest': False,
        'models': [
            {
                "class": LinearRegressionModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": MedianBaseModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": LightGBMModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": RandomForestModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": GradientBoosting,
                "use_categorical": True,
                "feature_type": "regression"
            }
        ]
    },
    {
        'name': 'tabler/tabler',
        'source_directory': 'src',
        'get_newest': False,
        'models': [
            {
                "class": LinearRegressionModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": MedianBaseModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": LightGBMModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": RandomForestModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": GradientBoosting,
                "use_categorical": True,
                "feature_type": "regression"
            }
        ]
    },
    {
        'name': 'google/material-design-lite',
        'source_directory': 'src',
        'get_newest': False,
        'models': [
            {
                "class": LinearRegressionModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": MedianBaseModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": LightGBMModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": RandomForestModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": GradientBoosting,
                "use_categorical": True,
                "feature_type": "regression"
            }
        ]
    },
    {
        'name': 'google/gson',
        'source_directory': '',
        'get_newest': False,
        'models': [
            {
                "class": LinearRegressionModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": MedianBaseModel,
                "use_categorical": False,
                "feature_type": "regression"
            },
            {
                "class": LightGBMModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": RandomForestModel,
                "use_categorical": True,
                "feature_type": "regression"
            },
            {
                "class": GradientBoosting,
                "use_categorical": True,
                "feature_type": "regression"
            }
        ]
     },
]
