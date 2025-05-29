import pandas as pd
from lifelines import CoxTimeVaryingFitter
from lifelines.utils import concordance_index

from src.predictions.base_model import BaseModel


class CoxTimeVaryingFitterModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = CoxTimeVaryingFitter()

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        exclude = {'path', 'start', 'stop', 'event'}
        return [c for c in df.columns if c not in exclude]

    def train(self, x_train, y_train):
        df_train = x_train.copy()
        df_train[['start', 'stop', 'event']] = y_train[['start', 'stop', 'event']]

        feat_cols = self._get_feature_columns(df_train)

        df_train[feat_cols] = self.scaler.fit_transform(df_train[feat_cols])

        self.model.fit(df_train, id_col='path', event_col='event', start_col='start', stop_col='stop')
        self.logger.info("CoxTimeVaryingFitter trained successfully.")

    def evaluate(self, x_test, y_test):
        df_test = x_test.copy()
        df_test[['start', 'stop', 'event']] = y_test[['start', 'stop', 'event']]

        feat_cols = self._get_feature_columns(df_test)
        df_test[feat_cols] = self.scaler.transform(df_test[feat_cols])

        ph = self.model.predict_partial_hazard(df_test)
        last_ph = ph.groupby(df_test['path']).last()

        # gather true times and events per subject
        times = y_test.groupby(x_test['path'])['stop'].last()
        events = y_test.groupby(x_test['path'])['event'].last()

        # concordance (negate hazard: higher hazard = shorter survival)
        cindex = concordance_index(times, -last_ph, events)
        self.logger.info(f"Time-varying Cox C-index: {cindex:.4f}")

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
