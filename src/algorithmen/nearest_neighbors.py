from typing import List

from src.algorithmen.metric import get_distance


class RateObject:
    def __init__(self):
        self.testedData = 0
        self.truePosData = 0
        self.trueNegData = 0
        self.falsePosData = 0
        self.falseNegData = 0

    def get_tp_ratio(self):
        return self.truePosData / (self.truePosData + self.falseNegData)

    def get_tn_ratio(self):
        return self.trueNegData / (self.trueNegData + self.falsePosData)

    def get_fp_ratio(self):
        return self.falsePosData / (self.falsePosData + self.trueNegData)

    def get_fn_ratio(self):
        return self.falseNegData / (self.falseNegData + self.truePosData)

    def get_f_ratio(self):
        return (self.falseNegData + self.falsePosData) / self.testedData


class Neighbour:
    def __init__(self, features, clazz):
        self.features = features
        self.clazz = clazz
        self.distance = 0

    def calc_distance(self, to, metric_name):
        self.distance = get_distance(self.features, to, metric_name)


class NearestNeighbour(RateObject):
    known_data_array: List[Neighbour]

    def __init__(self, k, metric_name):
        RateObject.__init__(self)
        self.metric_name = metric_name
        self.k = k
        self.known_data_array = []

    def train_data(self, data, clazz):
        self.known_data_array.append(Neighbour(data, clazz))

    def get_nearest_neighbours(self):
        for known_data in self.known_data_array:
            known_data.calc_distance()

    def predict_data(self, inputs, expected):
        for known_data in self.known_data_array:
            known_data.calc_distance(inputs, self.metric_name)
        self.known_data_array.sort(key=lambda x: x.distance)
        positives = 0
        negatives = 0
        if self.k > len(self.known_data_array):
            self.k = len(self.known_data_array)
        for i in range(self.k):
            if self.known_data_array[i].clazz == 1:
                positives += 1
            else:
                negatives += 1
        if positives > negatives:
            predicted = 1
        else:
            predicted = 0

        self.testedData += 1
        if expected == 1:
            if predicted == 1:
                self.truePosData += 1
            else:
                self.falseNegData += 1
        else:
            if predicted == 0:
                self.trueNegData += 1
            else:
                self.falsePosData += 1
