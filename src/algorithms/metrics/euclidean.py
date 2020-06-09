from algorithms.data_point import DataPoint
from algorithms.metrics.metric import Metric


class Euclidean(Metric):
    def get_distance(self, point_a: DataPoint, point_b: DataPoint) -> float:
        distance = 0
        for i in range(0, len(point_a.features)):
            distance += pow((abs(point_a.features[i] - point_b.features[i])), 2)
        return pow(distance, 1 / 2)
