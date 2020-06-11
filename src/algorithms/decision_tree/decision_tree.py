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
        # self.split_leaf(leaf)
        # feature_list = []
        # classes_list = []
        # for data_input in data_array:
        #    feature_list.append(data_input.features)
        #    classes_list.append(data_input.class_expected.value)
        # self.tree.fit(feature_list, classes_list)

    def predict_data(self, data_point: DataPoint) -> Classification:
        return data_point.class_expected
        # prediction_array = self.tree.predict([data_point.features])
        # prediction_value = prediction_array[0]
        # prediction_class = self.to_classification(prediction_value)
        # self.update_rates(data_point.class_expected, prediction_class)
        # return prediction_class

    def split_leaf(self, leaf: Leaf):
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
        return new_node

    def print_tree(self):
        print(self.root.print_node())
