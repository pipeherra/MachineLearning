import math
import numpy as np
import pandas as pd


class Statistics:
    @staticmethod
    def get_mean(signals):
        mean = 0.0
        for signal in signals:
            mean += signal
        mean /= len(signals)
        return mean

    @staticmethod
    def get_moving_average(signals, window):
        averages = []
        indexes = []
        if window % 2 == 0:
            print("Signal-Window is even: {}".format(window))
            return averages
        index = int((window - 1) / 2)
        for i in range(len(signals) - window + 1):
            average = 0.0
            for j in range(window):
                average += signals[i + j]
            average /= window
            averages.append(average)
            indexes.append(index)
            index = index + 1
        return pd.Series(averages, indexes)

    @staticmethod
    def get_median(signals):
        signals = np.sort(signals)
        signals_len = len(signals)
        if signals_len % 2 == 0:
            middle = int(signals_len / 2)
            median = (signals[middle - 1] + signals[middle]) / 2
        else:
            middle = int((signals_len - 1) / 2)
            median = signals[middle]
        return median

    @staticmethod
    def get_standard_deviation(signals, delta=1):
        mean = Statistics.get_mean(signals)
        standard_deviation = 0.0
        for signal in signals:
            standard_deviation += math.pow(signal - mean, 2)
        standard_deviation /= (len(signals) - delta)
        standard_deviation = math.sqrt(standard_deviation)
        return standard_deviation

    @staticmethod
    def get_timestamps_normalized(timestamps):
        time0 = timestamps.min()
        timestamps_normalized = []
        for timestamp in timestamps:
            timestamps_normalized.append(timestamp - time0)
        return timestamps_normalized

    @staticmethod
    def get_len(timestamps):
        time_min = timestamps.min()
        time_max = timestamps.max()
        return time_max - time_min
