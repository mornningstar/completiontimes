import logging
import os
import threading

import pandas as pd

csv_lock = threading.Lock()

def append_to_master_results(result_data: dict, master_results_path: str):
    """
    Safely appends a new result row to the master CSV file.
    Creates the file and header if it doesn't exist.
    """
    with csv_lock:
        file_exists = os.path.exists(master_results_path)

        desired_headers = [
            'project',
            'model',
            'split_strategy',
            'configuration',
            'timestamp',
            'metrics'
        ]

        formatted_data = {
            'project': result_data.get('project'),
            'model': result_data.get('model'),
            'split_strategy': result_data.get('split_strategy'),
            'configuration': result_data.get('configuration'),
            'timestamp': result_data.get('timestamp'),
            'metrics': result_data.get('metrics')
        }

        result_df = pd.DataFrame([formatted_data], columns=desired_headers)

        try:
            if not file_exists:
                result_df.to_csv(master_results_path, index=False, mode='w', header=True)
            else:
                result_df.to_csv(master_results_path, index=False, mode='a', header=False)
        except Exception as e:
            logging.error(f"Failed to append to master results CSV: {e}")