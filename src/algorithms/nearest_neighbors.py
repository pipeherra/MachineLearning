from typing import List

import numpy as np

from algorithms.algorithm import Algorithm
from misc.classification import Classification
from misc.data_point import DataPoint
from metrics.metric import Metric


class Neighbour(DataPoint):

    def __init__(self, features, class_expected: Classification):
        super().__init__(features, class_expected)
        self.distance = 0

    def calc_distance(self, to: DataPoint, metric: Metric):
        self.distance = metric.get_distance(self, to)


class NearestNeighbour(Algorithm):
    known_data_array: List[Neighbour]

    def __init__(self, k: int, metric: Metric, classifications: List[Classification]):
        super().__init__(classifications)
        self.metric = metric
        self.k = k
        self.known_data_array = []

    def train_data(self, data_array: List[DataPoint]):
        for data in data_array:
            self.known_data_array.append(Neighbour(data.features, data.class_expected))

    def predict_data(self, data_point: DataPoint) -> Classification:
        for known_data in self.known_data_array:
            known_data.calc_distance(data_point, self.metric)
        self.known_data_array.sort(key=lambda x: x.distance)
        class_counts = np.zeros(len(self.classifications))
        if self.k > len(self.known_data_array):
            self.k = len(self.known_data_array)
        for data_index in range(self.k):
            for class_index in range(len(self.classifications)):
                if self.known_data_array[data_index].class_expected == self.classifications[class_index]:
                    class_counts[class_index] += 1
                    break
        predicted = self.classifications[class_counts.argmax()]
        self.update_rates(data_point.class_expected, predicted)
        return predicted
