from abc import ABC, abstractmethod


class Transfer(ABC):

    @abstractmethod
    def transfer(self, data: float) -> float:
        return 0.0
