from stats.statistic import Statistic


class Median(Statistic):
    def get_value(self, inputs):
        return Statistic.get_median(inputs)