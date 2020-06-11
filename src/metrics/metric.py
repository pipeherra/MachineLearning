from abc import ABC, abstractmethod

from misc.data_point import DataPoint


class Metric(ABC):
    @abstractmethod
    def get_distance(self, point_a: DataPoint, point_b: DataPoint) -> float:
        return 0

    def check_feature_len(self, point_a: DataPoint, point_b: DataPoint):
        if len(point_a.features) != len(point_b.features):
            raise AttributeError("Len of point_a and point_b is unequal! point_a={}, point_b={}"
                                 .format(len(point_a.features), len(point_b.features)))
