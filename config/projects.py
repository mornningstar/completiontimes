from src.predictions.prophet_model import ProphetModel
from src.predictions.statistical_predictions.seasonal_arima_base import SeasonalARIMABase

PROJECTS = [
    {
        'name': 'khoj-ai/khoj',
        'models': [ProphetModel],
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
                #'files': ['src/khoj/database/adapters/__init__.py'],
                'cluster': True,
                'horizon': 30,
                'threshold': 0.5,
                'consecutive_days': 7
            }
        },
    },
]
