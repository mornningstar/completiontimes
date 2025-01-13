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
        }
        #'name': 'mozilla/addons-server',
    },
    #{
        #'name': 'openedx/edx-platform'
    #}
]
