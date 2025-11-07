from abc import ABC, abstractmethod
import pandas as pd

class BaseFeatureEngineer(ABC):
    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method takes a raw dataframe and returns an engineered dataframe.
        Every dataset-specific feature engineer must implement this method.
        """
        pass
