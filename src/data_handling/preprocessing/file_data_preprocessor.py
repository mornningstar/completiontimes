from src.data_handling.preprocessing.file_data_handling import handle_gaps

import logging
import pandas as pd

class FileDataPreprocessor:
    def __init__(self, raw_target: str):
        self.raw_target = raw_target
        self.logger = logging.getLogger(self.__class__.__name__)

    def reindex_file_data(self, file_data: pd.DataFrame, full_date_range: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Reindex the given file_data to the provided full_date_range.
        """
        file_data = file_data.reindex(full_date_range)
        return file_data

    def process_file(self, file_data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies gap handling to the raw_target column.
        """
        # Apply gap handling on the raw target
        file_data[self.raw_target] = handle_gaps(file_data[self.raw_target], self.raw_target)
        # Optionally fill initial missing values with 0
        first_valid = file_data[self.raw_target].first_valid_index()
        if first_valid is not None:
            file_data.loc[file_data.index < first_valid, self.raw_target] = 0
        return file_data