from misc.sensor import Sensor
from stats.statistic import Statistic


class FeatureConfig:
    def __init__(self, column: str, statistic: Statistic, sensor: Sensor = None):
        self.column = column
        self.statistic = statistic
        self.sensor = sensor
