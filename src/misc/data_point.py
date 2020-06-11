from typing import List

from misc.classification import Classification


class DataPoint:
    def __init__(self, features: List, class_expected: Classification):
        self.features = features
        self.class_expected = class_expected
