from src.predictions.ensemble_methods.random_forest import RandomForestModel
from src.predictions.lightgbm_model import LightGBMModel

PROJECTS = [
    # {
    #     'name': 'flairNLP/fundus',
    #     'source_directory': 'src',
    #     'get_newest': False,
    #     'models': [LightGBMModel],
    # },
    # {
    #     'name': 'khoj-ai/khoj',
    #     'source_directory': 'src',
    #     'get_newest': False,
    #     'models': [LightGBMModel]
    # },
    # {
    #     'name': 'vuejs/core',
    #     'source_directory': 'packages',
    #     'get_newest': False,
    #     'models': [LightGBMModel],
    # },
    # {
    #     'name': 'mozilla/addons-server',
    #     'source_directory': 'src',
    #     'get_newest': False,
    #     'models': [RandomForestModel],
    # },
    {
        'name': 'fastapi/fastapi',
        'source_directory': 'fastapi',
        'get_newest': True,
        'models': [
            {"class": RandomForestModel, "use_categorical": False},
        ]
    },
    {
        'name': 'pallets/flask',
        'source_directory': 'src',
        'get_newest': True,
        'models': [],
    },
    {
        'name': 'keras-team/keras',
        'source_directory': 'keras',
        'get_newest': True,
        'models': [],
    },
    {
        'name': 'tabler/tabler',
        'source_directory': 'src',
        'get_newest': True,
        'models': [],
    },
    {
        'name': 'google/material-design-lite',
        'source_directory': 'src',
        'get_newest': True,
        'models': [],
    },
    {
        'name': 'airbnb/kaldb',
        'source_directory': 'astra',
        'get_newest': True,
        'models': [],
    },
    {
        'name': 'slackhq/astra',
        'source_directory': 'astra',
        'get_newest': True,
        'models': [],
    },
    {
        'name': 'google/gson',
        'source_directory': '',
        'get_newest': True,
        'models': [],
    },
]
