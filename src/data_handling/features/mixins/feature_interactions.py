class FeatureInteractionsMixin:
    def _add_feature_interactions(self, df):
        df["commits_x_growth"] = df["total_commits"] * df["recent_growth_ratio"]
        df["interval_x_entropy"] = df["avg_commit_interval"] * df["interval_entropy"]
        df["contrib_x_entropy"] = df["recent_contribution_ratio"] * df["interval_entropy"]
        df["average_growth_commit"] = df["cumulative_size"] / df["total_commits"]
        df["committer_x_interval_entropy"] = df["committer_entropy"] * df["interval_entropy"]

        return df