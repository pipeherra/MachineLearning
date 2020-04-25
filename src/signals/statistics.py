import math
import numpy

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
        if window % 2 == 0:
            print("Signal-Window is even: {}".format(window))
            return averages
        for i in range(len(signals) - window + 1):
            average = 0.0
            for j in range(window):
                average += signals[i + j]
            average /= window
            averages.append(average)
        return averages

    @staticmethod
    def get_median(signals):
        signals = numpy.sort(signals)
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
