import math
from abc import ABC, abstractmethod
import numpy as np


class Statistic(ABC):
    @staticmethod
    def get_mean(inputs) -> float:
        mean = 0.0
        for signal in inputs:
            mean += signal
        mean /= len(inputs)
        return mean

    @staticmethod
    def get_median(inputs) -> float:
        inputs = np.sort(inputs)
        signals_len = len(inputs)
        if signals_len % 2 == 0:
            middle = int(signals_len / 2)
            median = (inputs[middle - 1] + inputs[middle]) / 2
        else:
            middle = int((signals_len - 1) / 2)
            median = inputs[middle]
        return median

    @staticmethod
    def get_standard_deviation(inputs, delta=1) -> float:
        mean = Statistic.get_mean(inputs)
        standard_deviation = 0.0
        for signal in inputs:
            standard_deviation += math.pow(signal - mean, 2)
        standard_deviation /= (len(inputs) - delta)
        standard_deviation = math.sqrt(standard_deviation)
        return standard_deviation

    @abstractmethod
    def get_value(self, inputs) -> float:
        pass
