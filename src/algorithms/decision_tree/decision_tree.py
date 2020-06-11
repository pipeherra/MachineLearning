from typing import List, Union

from algorithms.algorithm import Algorithm
from misc.classification import Classification
from misc.data_point import DataPoint
from algorithms.decision_tree.inode import Inode
from algorithms.decision_tree.leaf import Leaf
from algorithms.decision_tree.node import Node


class DecisionTree(Algorithm):
    root: Union[None, Node]
    window_count: int
    theta: float

    def __init__(self, classifications: List[Classification], theta: float, window_count: int):
        super().__init__(classifications)
        self.theta = theta
        self.window_count = window_count
        self.root = None

    def train_data(self, data_array: List[DataPoint]):
        self.root = Leaf(data_array)
        self.root = self.split_leaf(self.root)

    def predict_data(self, data_point: DataPoint) -> Classification:
        prediction_class = self.get_classification(self.root, data_point.features)
        self.update_rates(data_point.class_expected, prediction_class)
        return prediction_class

    def get_classification(self, node: Node, features: List) -> Classification:
        if isinstance(node, Leaf):
            return node.classification
        if isinstance(node, Inode):
            if features[node.split_index] < node.split_value:
                return self.get_classification(node.child_true, features)
            return self.get_classification(node.child_false, features)
        raise AttributeError("Unknown class")

    def split_leaf(self, leaf: Leaf):
        if leaf.entropy < self.theta:
            return leaf
        new_node: Union[Inode, None] = None
        for feature_index in range(leaf.feature_len):
            feature = []
            for data_point in leaf.data:
                feature.append(data_point.features[feature_index])
            feature_max = max(feature)
            feature_min = min(feature)
            feature_range = feature_max - feature_min
            feature_window_size = feature_range / self.window_count
            for window_index in range(self.window_count):
                window_max = feature_min + window_index * feature_window_size
                data_true = []
                data_false = []
                for data_point in leaf.data:
                    if data_point.features[feature_index] < window_max:
                        data_true.append(data_point)
                    else:
                        data_false.append(data_point)
                if len(data_true) == 0 or len(data_false) == 0:
                    continue
                leaf_true = Leaf(data_true)
                leaf_false = Leaf(data_false)
                temp_node = Inode(feature_index, window_max, leaf_true, leaf_false)
                if new_node is None or new_node.entropy > temp_node.entropy:
                    new_node = temp_node
                if new_node.entropy < self.theta:
                    return new_node
        new_node.child_false = self.split_leaf(new_node.child_false)
        new_node.child_true = self.split_leaf(new_node.child_true)
        return new_node

    def print_tree(self):
        print(self.root.print_node())
