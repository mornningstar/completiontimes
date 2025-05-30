class ChangeQualityFeatureMixin:
    def _add_change_quality_features(self, df):
        df["add_ratio"] = (
                df["lines_added"] / df["line_change"].replace(0, np.nan)
        ).fillna(0).clip(0, 1)

        df["pure_addition"] = ((df["lines_added"] > 0) & (df["lines_deleted"] == 0)).astype(int)
        df["pure_deletion"] = ((df["lines_deleted"] > 0) & (df["lines_added"] == 0)).astype(int)

        df["pure_addition_count"] = df.groupby("path")["pure_addition"].cumsum()
        df["pure_deletion_count"] = df.groupby("path")["pure_deletion"].cumsum()

        return df