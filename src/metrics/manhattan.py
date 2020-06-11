from misc.data_point import DataPoint
from metrics.metric import Metric


class Manhattan(Metric):
    def get_distance(self, point_a: DataPoint, point_b: DataPoint) -> float:
        distance = 0.0
        for i in range(0, len(point_a.features)):
            distance += abs(point_a.features[i] - point_b.features[i])
        return distance
