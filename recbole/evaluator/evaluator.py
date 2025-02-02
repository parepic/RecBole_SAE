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

class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics."""

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config["metrics"]]
        self.metric_class = {}

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

    def evaluate(self, dataobject: DataStruct):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``

        """
        result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            result_dict.update(metric_val)
        return result_dict
    
    
    def evaluate_fairness(self, recommendation_count):
        """
        Evaluate Long Tail Coverage and Coverage.
        
        Args:
            recommendation_count (list): Array where index `i` represents the count for item with ID `i`.
            csv_file_path (str): Path to the CSV file containing 'item_id' and 'popularity_label'.
            
        Returns:
            dict: A dictionary with 'Long Tail Coverage' and 'Coverage' metrics.
        """
        # Load the CSV file for item metadata
        item_data = pd.read_csv(r'./dataset/ml-1m/item_popularity_labels_with_titles.csv')
        
        # Determine the maximum possible number of items
        num_items = len(recommendation_count)
        
        # Filter items within the valid range [0, len(recommendation_count)-1]
        item_data = item_data[item_data['item_id:token'] < num_items]
        
        # Identify long-tail items (popularity_label == -1)
        long_tail_items = set(item_data[item_data['popularity_label'] != 1 ]['item_id:token'])
        
        # Items that were recommended at least once
        recommended_items = {i for i, count in enumerate(recommendation_count) if count > 0}
        
        # Calculate Long Tail Coverage
        recommended_long_tail_items = recommended_items & long_tail_items
        long_tail_coverage = len(recommended_long_tail_items) / len(long_tail_items) if long_tail_items else 0
        
        # Calculate Coverage (based on all possible items)
        coverage = len(recommended_items) / num_items
        
        # Return the metrics
        return long_tail_coverage, coverage