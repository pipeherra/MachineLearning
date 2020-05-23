import numpy as np
from src.algorithmen.metric import get_distance


def k_nearest_neighbours(data, inputs, metric_name, k):
    neighbours = []
    distance_k = 0
    for x in data:
        distance = get_distance(inputs, x[0:len(x) - 2], metric_name)
        if len(neighbours) < k:
            neighbours += [[x, distance]]
            distance_k = max(distance_k, distance)
        elif distance < distance_k:
            for y in neighbours:
                if y[1] >= distance_k:
                    neighbours.remove(y)
                    break
            neighbours += [[x, distance]]
            distance_k = distance
            for y in neighbours:
                distance_k = max(distance_k, y[1])
    class_1 = 0
    class_0 = 0
    for x in neighbours:
        if x[0][len(x[0]) - 2] == 1:
            class_1 += 1
        else:
            class_0 += 1
    if class_1 > class_0:
        result = 1
    else:
        result = 0
    return result


class object:
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


class nearest_neighbour(object):
    def __init__(self, k, metric_name):
        object.__init__(self)
        self.metric_name = metric_name
        self.k = k
        self.data = []

    def train_data(self, input, result):
        self.data += [[input, result]]

    def test_data(self, inputs, result):
        neighbour_result = k_nearest_neighbours(self.data, inputs, self.metric_name, self.k)

        self.testedData += 1
        if result == 1:
            if np.round(neighbour_result) == 1:
                self.truePosData += 1
            else:
                self.falseNegData += 1
        else:
            if np.round(neighbour_result) == 0:
                self.trueNegData += 1
            else:
                self.falsePosData += 1
