from src.predictions.ensemble_methods.random_forest import RandomForestModel
from src.predictions.prophet_model import ProphetModel
from src.predictions.statistical_predictions.seasonal_arima_base import SeasonalARIMABase

PROJECTS = [
    {
       'name': 'khoj-ai/khoj',
       'models': [RandomForestModel],
       "recluster": False,
        "replot": False,
        "plot_options": {
           'hierarchical': True,

           'cooccurrence_matrix': True,
           'cooccurrence_data': 'raw',  # Choose 'raw' or 'categorised'
            'top_n_files': 10,  # Change the number of top files for the co-occurrence matrix

            'proximity_matrix': True,
            'proximity_histogram': False,  # Skip the histogram

            'distance_vs_cooccurrence': True,
            'distance_vs_cooccurrence_data': 'scaled',  # Choose 'raw' or 'scaled'

            'zipf_distribution': True
        },
        'file_modeling_tasks': {
           'cluster_cumulative_size': {
                'files': ['src/khoj/database/adapters/__init__.py'],
                'cluster': True,
                'horizon': 90,
                'threshold': 0.1,
                'consecutive_days': 14
            }
        },
    },
    # {
    #     'name': 'openedx/edx-platform',
    #     'models': [ProphetModel],
    #     'recluster': True,
    #     'replot': True,
    #     'plot_options': {
    #         'hierarchical': True,
    #
    #         'cooccurrence_matrix': True,
    #         'cooccurrence_data': 'raw',  # Choose 'raw' or 'categorised'
    #         'top_n_files': 10,  # Change the number of top files for the co-occurrence matrix
    #
    #         'proximity_matrix': True,
    #         'proximity_histogram': True,  # Skip the histogram
    #
    #         'distance_vs_cooccurrence': True,
    #         'distance_vs_cooccurrence_data': 'scaled',  # Choose 'raw' or 'scaled'
    #
    #         'zipf_distribution': True
    #     },
    #     'file_modeling_tasks': {
    #         'cluster_cumulative_size': {
    #             'cluster': True,
    #             'horizon': 90,
    #             'threshold': 0.1,
    #             'consecutive_days': 14
    #         }
    #     }
    # },
    # # {
    # #     'name': 'mozilla/addons-server',
    # #     'models': [ProphetModel],
    # #     'recluster': True,
    # #     'replot': True,
    # #     'plot_options': {
    # #         'hierarchical': True,
    # #
    # #         'cooccurrence_matrix': True,
    # #         'cooccurrence_data': 'raw',  # Choose 'raw' or 'categorised'
    # #         'top_n_files': 10,  # Change the number of top files for the co-occurrence matrix
    # #
    # #         'proximity_matrix': True,
    # #         'proximity_histogram': True,  # Skip the histogram
    # #
    # #         'distance_vs_cooccurrence': True,
    # #         'distance_vs_cooccurrence_data': 'scaled',  # Choose 'raw' or 'scaled'
    # #
    # #         'zipf_distribution': True
    # #     },
    # #     'file_modeling_tasks': {
    # #         'cluster_cumulative_size': {
    # #             'cluster': True,
    # #             'horizon': 90,
    # #             'threshold': 0.1,
    # #             'consecutive_days': 14
    # #         }
    # #     }
    # # }
]
