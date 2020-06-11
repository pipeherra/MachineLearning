from stats.statistic import Statistic


class StandardDeviation(Statistic):
    def get_value(self, inputs):
        return Statistic.get_standard_deviation(inputs)