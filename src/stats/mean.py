from stats.statistic import Statistic


class Mean(Statistic):
    def get_value(self, inputs):
        return Statistic.get_mean(inputs)
