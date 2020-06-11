from abc import ABC, abstractmethod
from typing import Union


class Node(ABC):
    parent: Union['Node', None]

    def __init__(self):
        self.parent = None
        self.entropy = 0.0
        self.data_count = 0

    def get_entropy(self) -> float:
        return self.entropy

    def get_data_count(self) -> int:
        return self.data_count

    @abstractmethod
    def analyze_node(self):
        pass

    @abstractmethod
    def print_node(self, level: int = 0) -> str:
        pass
