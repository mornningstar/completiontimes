from src.predictions.statistical_predictions.seasonal_arima_base import SeasonalARIMABase

PROJECTS = [
    {
        'name': 'khoj-ai/khoj',
        'models': [SeasonalARIMABase],
        'file_modeling_tasks': {
            'size': {
                'files': ['src/interface/emacs/khoj.el', 'src/khoj/database/adapters/__init__.py'],
                'cluster': True
            },
            'cumulative_size': {
                'size': {
                'files': ['src/interface/emacs/khoj.el', 'src/khoj/database/adapters/__init__.py'],
                'cluster': True
                }
            }
        },
        'general_config': {
            'historical_validation_split': 0.8,  # Define split ratio for historical validation
            'time_horizon': 30,  # Define prediction horizon (e.g., for no-commit probability)
            'auto_tune': True,  # Enable/disable auto-tuning for models
        }
    },
]
