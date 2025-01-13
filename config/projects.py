from src.predictions.statistical_predictions.seasonal_arima_base import SeasonalARIMABase

PROJECTS = [
    {
        'name': 'khoj-ai/khoj',
        'models': [SeasonalARIMABase],
        "recluster": False,
        "replot": True,
        "plot_options": {
            'hierarchical': True,

            'cooccurrence_matrix': True,
            'cooccurrence_data': 'categorised',  # Choose 'raw' or 'categorized'
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
                'cluster': True
            },
            'cumulative_size': {
                'size': {
                'files': ['src/interface/emacs/khoj.el', 'src/khoj/database/adapters/__init__.py'],
                'cluster': True
                }
            }
        },
    },
]
