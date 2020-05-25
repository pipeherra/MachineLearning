from typing import List

from src.algorithmen.metric import get_distance
import numpy as np


class RateObject:
    def __init__(self):
        self.testedData = 0
        self.truePosData = 0
        self.trueNegData = 0
        self.falsePosData = 0
        self.falseNegData = 0
        self.correctClassifications = 0
        self.wrongClassifications = 0

    def get_tp_ratio(self):
        return self.truePosData / (self.truePosData + self.falseNegData)

    def get_tn_ratio(self):
        return self.trueNegData / (self.trueNegData + self.falsePosData)

    def get_fp_ratio(self):
        return self.falsePosData / (self.falsePosData + self.trueNegData)

    def get_fn_ratio(self):
        return self.falseNegData / (self.falseNegData + self.truePosData)

    def get_f_ratio(self):
        return self.wrongClassifications / self.testedData


class Neighbour:
    def __init__(self, features, clazz):
        self.features = features
        self.clazz = clazz
        self.distance = 0

    def calc_distance(self, to, metric_name):
        self.distance = get_distance(self.features, to, metric_name)


class NearestNeighbour(RateObject):
    known_data_array: List[Neighbour]

    def __init__(self, k, metric_name, classes=None):
        RateObject.__init__(self)
        if classes is None:
            classes = [0.0, 1.0]
        self.metric_name = metric_name
        self.k = k
        self.known_data_array = []
        self.classes = classes

    def train_data(self, data, clazz):
        self.known_data_array.append(Neighbour(data, clazz))

    def get_nearest_neighbours(self):
        for known_data in self.known_data_array:
            known_data.calc_distance()

    def predict_data(self, inputs, expected):
        for known_data in self.known_data_array:
            known_data.calc_distance(inputs, self.metric_name)
        self.known_data_array.sort(key=lambda x: x.distance)
        class_counts = np.zeros(len(self.classes))
        if self.k > len(self.known_data_array):
            self.k = len(self.known_data_array)
        for data_index in range(self.k):
            for class_index in range(len(self.classes)):
                if self.known_data_array[data_index].clazz == self.classes[class_index]:
                    class_counts[class_index] += 1
                    break
        predicted = self.classes[class_counts.argmax()]

        self.testedData += 1
        if predicted == expected:
            self.correctClassifications += 1
        else:
            self.wrongClassifications += 1
        if len(self.classes) == 2:
            if predicted == self.classes[0]:
                if expected == self.classes[0]:
                    self.trueNegData += 1
                else:
                    self.falseNegData += 1
            else:
                if expected == self.classes[1]:
                    self.truePosData += 1
                else:
                    self.falsePosData += 1
        return predicted
