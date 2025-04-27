# -*- encoding: utf-8 -*-
# @Time    :   2021/6/25
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

"""
recbole.evaluator.evaluator
#####################################
"""

from recbole.evaluator.register import metrics_dict
from recbole.evaluator.collector import DataStruct
from collections import OrderedDict
import pandas as pd
import numpy as np


class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics."""

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config["metrics"]]
        self.metric_class = {}

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

    def evaluate(self, dataobject: DataStruct, chunks=None):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``

        """
        result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject, chunks=chunks)
            result_dict.update(metric_val)
        return result_dict
    
    
    
    
    
def evaluate_fairness(self, recommendation_count):
    """
    Evaluate Long Tail Coverage, Coverage, Gini Coefficient, and Average Recommendation Popularity (ARP),
    excluding item ID 0 (unused index).
    
    Args:
        recommendation_count (list): Array where index `i` represents the count for item with ID `i`.
        
    Returns:
        dict: A dictionary with 'Long Tail Coverage', 'Coverage', 'Gini Coefficient', and 'ARP' metrics.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    file = r'./dataset/ml-1m/item_popularity_labels_with_titles.csv'
    # Load item metadata
    item_data = pd.read_csv(file)

    # Slice recommendation_count to ignore index 0
    recommendation_count = recommendation_count[1:]
    offset = 1  # Because IDs start at 1

    # Adjust number of items
    num_items = len(recommendation_count)

    # Filter valid item_data range
    item_data = item_data[item_data['item_id:token'] <= num_items]

    # Identify long-tail groups
    long_tail_items = set(item_data[item_data['popularity_label'] != 1]['item_id:token'])
    deep_long_tail_items = set(item_data[item_data['popularity_label'] == -1]['item_id:token'])

    # Items recommended at least once (excluding index 0)
    recommended_items = {i + offset for i, count in enumerate(recommendation_count) if count > 0}

    # Long Tail Coverage
    recommended_LT_items = recommended_items & long_tail_items
    recommended_deep_LT_items = recommended_items & deep_long_tail_items
    long_tail_coverage = len(recommended_LT_items) / len(long_tail_items) if long_tail_items else 0
    deep_long_tail_coverage = len(recommended_deep_LT_items) / len(deep_long_tail_items) if deep_long_tail_items else 0

    # Coverage
    coverage = len(recommended_items) / num_items

    # Gini Coefficient
    sorted_counts = np.sort(recommendation_count)
    n = len(sorted_counts)
    gini_coefficient = (
        (2 * np.sum((np.arange(1, n + 1) * sorted_counts))) / (n * np.sum(sorted_counts)) - (n + 1) / n
        if np.sum(sorted_counts) > 0 else 0
    )

    # Proposed ARP calculation
    # Normalize popularity by the maximum interaction count for better interpretability
    max_interactions = item_data['interaction_count'].max()
    item_data['normalized_popularity'] = item_data['interaction_count'] / max_interactions

    # Create a lookup dictionary for normalized popularity
    popularity_lookup = dict(zip(item_data['item_id:token'], item_data['normalized_popularity']))

    # Since we only have recommendation_count (not per-user top-K lists), 
    # weight the popularity by how often each item is recommended
    total_recommendations = sum(recommendation_count)
    if total_recommendations > 0:
        weighted_popularity_sum = sum(
            popularity_lookup.get(item_id, 0.0) * count
            for item_id, count in enumerate(recommendation_count, start=offset)
            if count > 0
        )
        arp = weighted_popularity_sum / total_recommendations
    else:
        arp = 0.0

    return {
        'LT_coverage@10': long_tail_coverage,
        'Deep_LT_coverage@10': deep_long_tail_coverage,
        'coverage@10': coverage,
        'Gini_coef@10': gini_coefficient,
        'ARP@10': float(arp)
    }