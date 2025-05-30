import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from lifelines.utils import concordance_index

from src.predictions.base_model import BaseModel


class CoxTimeVaryingFitterModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = CoxTimeVaryingFitter(penalizer=1.0, l1_ratio=0.1)

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        exclude = {'path', 'start', 'stop', 'event'}
        return [c for c in df.columns if c not in exclude]

    def train(self, x_train, y_train):
        df_train = x_train.copy()
        df_train[["start", "stop", "event"]] = y_train[["start", "stop", "event"]]

        feat_cols = self._get_feature_columns(df_train)

        df_train[feat_cols] = self.scaler.fit_transform(df_train[feat_cols])

        self.model.fit(df_train, id_col='path', event_col='event', start_col='start', stop_col='stop')
        self.logger.info("CoxTimeVaryingFitter trained successfully.")

    def evaluate(self, x_test, y_test):
        df = x_test.reset_index(drop=True).copy()
        df[['start', 'event', 'stop']] = y_test[['start', 'event', 'stop']].reset_index(drop=True)

        grouped = df.groupby('path')
        times = grouped['stop'].last()
        events = grouped['event'].last()

        ph = self.model.predict_partial_hazard(x_test).reset_index(drop=True)
        df['ph'] = ph
        last_ph = df.groupby('path')['ph'].last()

        times, last_ph = times.align(last_ph, join='inner')
        events = events.reindex(times.index)

        # concordance (negate hazard: higher hazard = shorter survival)
        cindex = concordance_index(times, -last_ph, events)

        return cindex

    def predict_risk_score(self, x: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
        """
        Predicts the partial hazard (risk score) per sample.
        """
        df_pred = x.copy()
        df_pred[['start', 'stop', 'event']] = y[['start', 'stop', 'event']]

        feat_cols = self._get_feature_columns(df_pred)
        df_pred[feat_cols] = self.scaler.transform(df_pred[feat_cols])

        return self.model.predict_partial_hazard(df_pred)

    def get_feature_importances(self) -> pd.Series:
        """
        Return absolute value of Cox coefficients sorted descending.
        """
        return self.model.params_.abs().sort_values(ascending=False)
