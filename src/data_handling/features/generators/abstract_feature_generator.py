from abc import ABC, abstractmethod

import pandas as pd


class AbstractFeatureGenerator(ABC):

    @abstractmethod
    def generate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """ Generates features and adds them to the dataframe. """
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """ Returns a list of feature names (or prefixes) created by this generator. """
        pass
