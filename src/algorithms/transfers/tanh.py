import math

from algorithms.transfers.transfer import Transfer


class Tanh(Transfer):
    def transfer(self, data: float) -> float:
        return (math.tanh(data) * 0.5) + 0.5
