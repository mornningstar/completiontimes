import logging

import numpy as np
import pandas as pd


class CompletionDateLabler:
    STABILITY_MIN_COMMITS = 3
    STABILITY_MIN_DAYS = 14
    STABILITY_IDLE_DAYS = 30
    STABILITY_PERCENTAGE_CHANGE = 0.15

    def __init__(self, config: dict = None):
        self.logging = logging.getLogger(self.__class__.__name__)
        self.now = pd.Timestamp.utcnow().normalize().tz_localize(None)

        self.min_commits = config.get('stability_min_commits', 3)
        self.min_days = config.get('stability_min_days', 14)
        self.idle_days = config.get('stability_idle_days', 30)
        self.percentage_change = config.get('stability_percentage_change', 0.15)
        self.percentile_cutoff = config.get('project_inactivity_cutoff_percentile', 0.95)

    def add_days_until_completion(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce").dt.tz_localize(None)

        df["days_until_completion"] = (
                df["completion_date"] - df["date"]
        ).dt.days
        df["days_until_completion"] = df["days_until_completion"].clip(lower=0)

        return df

    def label(self, df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, int, int, pd.Series]:
        """
        Add a 'completion_date' column for each file based on two strategies:
        1. A stable pattern: percentage_change stays below threshold for consecutive_days commits
        2. A deletion event:
        3. A long period of inactivity after the last commit (idle_days_cutoff)
        """
        df['completion_date'] = pd.NaT
        df['completion_reason'] = None

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)

        if "commit_interval_days" not in df.columns:
            raise ValueError("'commit_interval_days' must be calculated before calling this labler.")

        project_cutoff = df["commit_interval_days"].replace(0, np.nan).dropna().quantile(self.percentile_cutoff)
        project_cutoff = int(np.clip(project_cutoff, 30, 365))
        self.logging.info(f"Using project-wide inactivity cutoff of {project_cutoff} days")

        for path, group in df.groupby("path"):
            # Strategy 1
            completion_date, reason = self._check_stable_line_change_window(group)

            if completion_date:
                df.loc[df["path"] == path, "completion_date"] = pd.to_datetime(completion_date).tz_localize(None)
                df.loc[df["path"] == path, "completion_reason"] = reason
                continue

            # Strategy 2: Explicit deletion (size = 0)
            if group["size"].iloc[-1] == 0:
                deletion_date = group["date"].iloc[-1]
                df.loc[df["path"] == path, "completion_date"] = pd.to_datetime(deletion_date).tz_localize(None)
                df.loc[df["path"] == path, "completion_reason"] = "deleted"
                continue

            # Strategy 3: Inactivity fallback
            last_commit_date = group["date"].max()
            days_since_last_commit = (self.now - last_commit_date.tz_localize(None)).days

            if days_since_last_commit > project_cutoff:
                df.loc[df["path"] == path, "completion_date"] = last_commit_date.tz_localize(None)
                df.loc[df["path"] == path, "completion_reason"] = "idle_timeout"

        num_completed_files = df[df['completion_date'].notna()]['path'].nunique()
        total_files = df['path'].nunique()
        self.logging.info(
            f"Completed files: {num_completed_files} / {total_files} ({(num_completed_files / total_files * 100):.2f}%)")

        strategy_counts = (
            df[df['completion_reason'].notna()]
            .groupby("path")
            .first()["completion_reason"]
            .value_counts()
        )
        for reason, count in strategy_counts.items():
            self.logging.info(f"{reason}: {count} files")

        return df, num_completed_files, total_files, strategy_counts

    def _check_stable_line_change_window(self, group):
        group = group.sort_values("date").reset_index(drop=True)
        median_change = group["line_change"].median()
        threshold = max(3, median_change * self.percentage_change)

        if group.iloc[-1]["line_change"] > threshold:
            return None, None

        stable_block_indices = []
        for idx in range(len(group) - 1, -1, -1):
            if group.loc[idx, "line_change"] <= threshold:
                stable_block_indices.append(idx)
            else:
                break

        if not stable_block_indices:
            return None, None

        stable_block = group.loc[sorted(stable_block_indices)]
        if len(stable_block) < self.min_commits:
            return None, None

        block_duration = (stable_block["date"].iloc[-1] - stable_block["date"].iloc[0]).days
        if block_duration < self.min_days:
            return None, None

        final_commit_date = stable_block["date"].iloc[-1]
        if (self.now - final_commit_date).days < self.idle_days or group["size"].iloc[-1] == 0:
            return None, None

        return final_commit_date, "stable_line_change"