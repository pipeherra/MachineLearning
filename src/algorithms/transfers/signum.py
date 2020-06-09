from algorithms.transfers.transfer import Transfer


class Signum(Transfer):
    def transfer(self, data: float) -> float:
        if data < 0.0:
            return 0.0
        return 1.0
