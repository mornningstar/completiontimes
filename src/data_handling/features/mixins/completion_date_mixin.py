import logging

import numpy as np
import pandas as pd


class CompletionDateMixin:
    def __init__(self):
        self.logging = logging.getLogger(self.__class__.__name__)
        self.now = pd.Timestamp.utcnow().normalize().tz_localize(None)

    def add_days_until_completion(self, df):
        df = df.copy()

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce").dt.tz_localize(None)
        df.loc[df["completion_date"].notnull(), "completion_date"] = (
            df.loc[df["completion_date"].notnull(), "completion_date"].dt.tz_localize(None)
        )

        df["days_until_completion"] = (
                df["completion_date"] - df["date"]
        ).dt.days

        df["days_until_completion"] = df["days_until_completion"].clip(lower=0)

        return df

    def add_completion_labels(self, df):
        """
            Add a 'completion_date' column for each file based on two strategies:
            1. A stable pattern: percentage_change stays below threshold for consecutive_days commits
            2. A deletion event:
            3. A long period of inactivity after the last commit (idle_days_cutoff)
        """
        df['completion_date'] = pd.NaT
        df['completion_reason'] = None

        #df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)

        project_cutoff = df["commit_interval_days"].replace(0, np.nan).dropna().quantile(0.95)
        project_cutoff = int(np.clip(project_cutoff, 30, 365))
        self.logging.info(f"Using project-wide inactivity cutoff of {project_cutoff} days")

        for path, group in df.groupby("path"):
            # Strategy 1
            completion_date, reason = self._check_stable_line_change_window(group)

            if completion_date:
                completion_date = pd.to_datetime(completion_date).tz_localize(None)
                df.loc[df["path"] == path, "completion_date"] = completion_date
                df.loc[df["path"] == path, "completion_reason"] = reason
                continue

            # Strategy 2: Explicit deletion (size = 0)
            if group["size"].iloc[-1] == 0:
                deletion_date = group["date"].iloc[-1]
                deletion_date = pd.to_datetime(deletion_date).tz_localize(None)
                df.loc[df["path"] == path, "completion_date"] = deletion_date
                df.loc[df["path"] == path, "completion_reason"] = "deleted"
                continue

            # Strategy 3: Inactivity fallback
            last_commit_date = group["date"].max()
            last_commit_date = last_commit_date.tz_localize(None).to_pydatetime()
            days_since_last_commit = (self.now - last_commit_date).days

            if days_since_last_commit > project_cutoff:
                df.loc[df["path"] == path, "completion_date"] = last_commit_date
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

        return df, num_completed_files, total_files

    def _check_stable_line_change_window(self, group):
        group = group.sort_values("date").reset_index(drop=True)

        min_commits = 3
        min_days = 14
        confirm_idle_days = 30
        now = pd.Timestamp.now().normalize()

        median_change = group["line_change"].median()
        threshold = max(3, median_change * 0.15)

        if group.iloc[-1]["line_change"] > threshold:
            # If the very last commit isn't stable, there's no valid stable ending
            return None, None

        stable_block_indices = []

        for idx in range(len(group) - 1, -1, -1):
            if group.loc[idx, "line_change"] <= threshold:
                stable_block_indices.append(idx)
            else:
                break

        stable_block_indices = sorted(stable_block_indices)
        stable_block = group.loc[stable_block_indices]

        if len(stable_block) < min_commits:
            return None, None

        block_duration = (stable_block["date"].iloc[-1] - stable_block["date"].iloc[0]).days
        if block_duration < min_days:
            return None, None

        final_commit_date = stable_block["date"].iloc[-1]
        if (now - final_commit_date).days < confirm_idle_days:
            return None, None

        if group["size"].iloc[-1] == 0:
            return None, None

        return final_commit_date, "stable_line_change"
