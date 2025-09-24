from abc import ABC, abstractmethod

import pandas as pd


class AbstractFeatureGenerator(ABC):
    @abstractmethod
    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """ Returns a list of feature names (or prefixes) created by this generator. """
        pass

    @abstractmethod
    def generate(self, df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, list[str]]:
        """
        Generates features and adds them to the dataframe.
        Returns the dataframe and a list of binary columns (to not touch them in scaling).
        """
        pass
