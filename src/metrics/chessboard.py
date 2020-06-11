from misc.data_point import DataPoint
from metrics.metric import Metric


class Chessboard(Metric):
    def get_distance(self, point_a: DataPoint, point_b: DataPoint) -> float:
        self.check_feature_len(point_a, point_b)
        distance = 0
        for i in range(0, len(point_a.features)):
            distance = max(distance, abs(point_a.features[i] - point_b.features[i]))
        return distance
