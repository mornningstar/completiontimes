import pandas as pd
import logging

class ClusterAggregator:
    def __init__(self, raw_target: str, target_contains_cumulative: bool = False):
        """
        raw_target: The file-level column to aggregate, e.g., "size".
        target_contains_cumulative: If True, we compute a cumulative sum.
        """
        self.raw_target = raw_target
        self.target_contains_cumulative = target_contains_cumulative
        self.logger = logging.getLogger(self.__class__.__name__)


    def aggregate_cluster_features(self, cluster_time_series: dict) -> pd.DataFrame:
        """
        Given a dict of {cluster_id: DataFrame}, aggregates data by date.
        """
        aggregated_clusters_list = []

        for cluster_id, cluster_df in cluster_time_series.items():
            # Group by date and sum the raw_target
            daily_sum = cluster_df.groupby(cluster_df.index)[self.raw_target].sum()

            if self.target_contains_cumulative:
                daily_cum = daily_sum.cumsum()
                aggregated_cluster = pd.DataFrame({
                    'cluster_size': daily_sum,
                    'cluster_cumulative_size': daily_cum
                })
            else:
                aggregated_cluster = pd.DataFrame({
                    'cluster_aggregate': daily_sum
                })
            aggregated_cluster['cluster'] = cluster_id
            aggregated_clusters_list.append(aggregated_cluster)

        aggregated_clusters_df = pd.concat(aggregated_clusters_list).sort_index()
        return aggregated_clusters_df