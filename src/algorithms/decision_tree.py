import math
from typing import List
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from algorithms.algorithm import Algorithm
from algorithms.classification import Classification
from algorithms.data_point import DataPoint


class DecisionTree(Algorithm):

    def __init__(self, classifications: List[Classification], theta):
        super().__init__(classifications)
        self.theta = theta
        self.tree = DecisionTreeClassifier(criterion='entropy')

    def train_data(self, data_array: List[DataPoint]):
        feature_list = []
        classes_list = []
        for data_input in data_array:
            feature_list.append(data_input.features)
            classes_list.append(data_input.class_expected.value)
        self.tree.fit(feature_list, classes_list)

    def predict_data(self, data_point: DataPoint) -> Classification:
        prediction_array = self.tree.predict([data_point.features])
        prediction_value = prediction_array[0]
        prediction_class = self.to_classification(prediction_value)
        self.update_rates(data_point.class_expected, prediction_class)
        return prediction_class

    def get_entropy(self, data_array: List[DataPoint]) -> float:
        instances_per_class = np.zeros(self.classes_len)
        instances_total = 0
        for data_point in data_array:
            current_class = self.classifications.index(data_point.class_expected)
            instances_per_class[current_class] += 1
            instances_total += 1
        entropy = 0.0
        for instances in instances_per_class:
            probability = instances / instances_total
            if 0.0 < probability <= 1.0:
                temp_entropy = probability * math.log(probability)
                entropy -= temp_entropy
            else:
                print("probability wrong: {}".format(probability))
        return entropy
