import logging

import numpy as np
import pandas as pd

from src.data_handling.features.base_feature_engineer import BaseFeatureEngineer


class SurvivalFeatureEngineer(BaseFeatureEngineer):
    def __init__(self, file_repo, plotter, use_categorical):
        super().__init__(file_repo, plotter, use_categorical)

    @staticmethod
    def get_target_columns():
        return ["start", "stop", "event"]

    def engineer_features(self, df, window = 7, is_static: bool = False):
        df = super().engineer_features(df)

        if is_static:
            out = self._add_survival_targets_static(df)
        else:
            out = self._add_survival_targets_time_varying(df)

        # fill NaNs on the one you’re returning:
        nums = [c for c in out.select_dtypes(include="number").columns
                if c not in self.get_target_columns()]
        out[nums] = out[nums].fillna(0.0)

        return out

    def _add_survival_targets_static(self, df):
        """
            One row per path with columns:
               duration – days from first commit to observed end-point
               event    – 1 if a terminal mixins occurred, 0 if censored
        """
        snap_num = self.collapse_to_first_last(df, base_cols=["size"])
        snap_dates = (
            df.sort_values(["path", "date"])
            .groupby("path")
            .agg(
                first_commit_date=("date", "first"),
                last_commit_date=("date", "last"),
                completion_date=("completion_date", "last"),
                completion_reason=("completion_reason", "last")
            )
            .reset_index()
        )
        snap = snap_num.merge(snap_dates, on="path", how="inner")

        today = pd.Timestamp.utcnow().normalize()
        TERMINAL_REASONS = {"stable_line_change", "idle_timeout"}
        is_event = snap["completion_reason"].isin(TERMINAL_REASONS)
        end_date = snap["completion_date"].where(is_event, today)

        snap["duration"] = (end_date - snap["first_commit_date"]).dt.days.clip(lower=0)
        snap["event"] = is_event.astype(int)

        return snap


    def _add_survival_targets_time_varying(self, df):
        """
        Transforms each record into counting-process intervals for a time-dependent Cox model.
        :param df:
        :return:
        """
        today = pd.Timestamp.utcnow().normalize().tz_localize(None)
        df = df.sort_values(["path", "date"]).copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce")

        df["end_date"] = df["completion_date"].fillna(today)
        df["next_date"] = df.groupby("path")["date"].shift(-1)

        df["start_date"] = df["date"]
        df["stop_date"] = df["next_date"].fillna(df["end_date"])

        first_commit = df.groupby("path")["date"].transform('min')
        df["start"] = (df["start_date"] - first_commit).dt.days.clip(lower=0)
        df["stop"] = (df["stop_date"] - first_commit).dt.days.clip(lower=0)

        df["event"] = ((df["stop_date"] == df["completion_date"]) &
                       df["completion_reason"].isin(["stable_line_change", "idle_timeout"]))
        df["event"] = df["event"].astype(int)

        df.drop(columns=["start_date", "stop_date", "end_date", "next_date", "date", "completion_date"],
                inplace=True, errors="ignore")

        df = df[df["stop"] > df["start"]].copy()
        df.reset_index(drop=True, inplace=True)
        return df
