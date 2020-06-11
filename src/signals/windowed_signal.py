from typing import List

from misc.classification import Classification

from misc.data_point import DataPoint
from misc.feature_config import FeatureConfig
from misc.sensor import Sensor
from signals.signal import Signal


class WindowedSignal(Signal):

    def __init__(self, path: str, classification: Classification, start: int, end: int,
                 window_size: int, window_moving: int):
        super().__init__(path, classification, start, end)
        i = 0
        window_start = 0
        window_end = window_size
        windows = []
        while window_end <= self.data_len:
            windows.append(
                self.data[(self.data.Timestamp_normalized >= window_start) &
                          (self.data.Timestamp_normalized < window_end)])
            i += 1
            window_start = i * window_moving
            window_end = i * window_moving + window_size
        self.windows = windows

    def get_data_points(self, feature_configs: List[FeatureConfig]):
        data_points = []
        for window in self.windows:
            features = []
            feature_config: FeatureConfig
            for feature_config in feature_configs:
                if feature_config.sensor is not None:
                    data = window[window.ID == feature_config.sensor.id]
                else:
                    data = window
                data = data[feature_config.column]
                features.append(feature_config.statistic.get_value(data))
            data_points.append(DataPoint(features, self.classification))
        return data_points




