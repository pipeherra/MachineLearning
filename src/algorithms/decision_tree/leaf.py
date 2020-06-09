import math
from typing import List, Union

from algorithms.classification import Classification
from algorithms.data_point import DataPoint
from algorithms.decision_tree.node import Node


class Leaf(Node):
    feature_len: int
    classification: Union[Classification, None]
    data: List[DataPoint]

    def __init__(self, data: List[DataPoint]):
        super().__init__()
        self.data = data
        self.feature_len = len(data[0].features)
        self.classification = None
        self.analyze_node()

    def analyze_node(self):
        instances_per_class = {}
        self.data_count = 0
        max_instances_per_class = 0
        for data_point in self.data:
            current_instances = instances_per_class.get(data_point.class_expected, 0) + 1
            if current_instances > max_instances_per_class:
                max_instances_per_class = current_instances
                self.classification = data_point.class_expected
            instances_per_class[data_point.class_expected] = current_instances
            self.data_count += 1
        entropy = 0.0
        for instances in instances_per_class.values():
            probability = instances / self.data_count
            if 0.0 < probability <= 1.0:
                temp_entropy = probability * math.log(probability)
                entropy -= temp_entropy
            else:
                print("probability wrong: {}".format(probability))
        self.entropy = entropy

    def print_node(self) -> str:
        return "Leaf: Data: {}, Entropy: {}, Class: {}".format(self.data_count, self.entropy, self.classification.name)




