from src.predictions.statistical_predictions.seasonal_arima_base import SeasonalARIMABase

PROJECTS = [
    {
        'name': 'khoj-ai/khoj',
        'models': [],
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
            'size': {
                'files': ['src/interface/emacs/khoj.el', 'src/khoj/database/adapters/__init__.py'],
                'cluster': False,
                'horizon': 30, # Prediction horizon in days
                'threshold': 10, # Completion criterion threshold: 10% change
                'consecutive_days': 7 # Criterion must be met for 7 consecutive days
            },
            'cumulative_size': {
                'files': ['src/interface/emacs/khoj.el', 'src/khoj/database/adapters/__init__.py'],
                'cluster': False,
                'horizon': 30,
                'threshold': 10,
                'consecutive_days': 7
            }
        },
    },
    {
        'name': 'openedx/edx-platform',
        "recluster": True,
        "replot": True,

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
    }
]
