from abc import ABC, abstractmethod
from typing import List

from algorithms.data_point import DataPoint
from algorithms.classification import Classification


class Algorithm(ABC):

    def __init__(self, classifications: List[Classification]):
        self.classifications = classifications
        self.classes_len = len(classifications)
        self.prediction_total_data_count = 0
        self.truePosData = 0
        self.trueNegData = 0
        self.falsePosData = 0
        self.falseNegData = 0
        self.prediction_correct_data_count = 0
        self.prediction_wrong_data_count = 0

    @abstractmethod
    def train_data(self, data_array: List[DataPoint]):
        pass

    @abstractmethod
    def predict_data(self, data_point: DataPoint) -> Classification:
        pass

    def get_tp_ratio(self) -> float:
        return self.truePosData / (self.truePosData + self.falseNegData)

    def get_tn_ratio(self) -> float:
        return self.trueNegData / (self.trueNegData + self.falsePosData)

    def get_fp_ratio(self) -> float:
        return self.falsePosData / (self.falsePosData + self.trueNegData)

    def get_fn_ratio(self) -> float:
        return self.falseNegData / (self.falseNegData + self.truePosData)

    def get_wrong_ratio(self) -> float:
        return self.prediction_wrong_data_count / self.prediction_total_data_count

    def get_correct_ratio(self) -> float:
        return self.prediction_correct_data_count / self.prediction_total_data_count

    def update_rates(self, class_expected: Classification, class_predicted: Classification):
        self.prediction_total_data_count += 1
        if class_expected == class_predicted:
            self.prediction_correct_data_count += 1
        else:
            self.prediction_wrong_data_count += 1
        if len(self.classifications) == 2:
            if class_predicted == self.classifications[0]:
                if class_expected == self.classifications[0]:
                    self.trueNegData += 1
                else:
                    self.falseNegData += 1
            else:
                if class_expected == self.classifications[1]:
                    self.truePosData += 1
                else:
                    self.falsePosData += 1

    def to_classification(self, prediction_value: float) -> Classification:
        for classification in self.classifications:
            if classification.value == prediction_value:
                return classification
        return Classification(prediction_value, "unknown")

    def print_statistics(self):
        print("Predictions total:  \t{},\t{}%".format(self.prediction_total_data_count, 100))
        print("Predictions correct:\t{},\t{}%".format(self.prediction_correct_data_count, self.get_correct_ratio() * 100))
        print("Predictions wrong:  \t{},\t{}%".format(self.prediction_wrong_data_count, self.get_wrong_ratio() * 100))
        if len(self.classifications) == 2:
            print("\nPositive = {}, Negative = {}".format(self.classifications[1].name, self.classifications[0].name))
            print("True  Positive:\t{},\t{}%".format(self.truePosData, self.get_tp_ratio() * 100))
            print("True  Negative:\t{},\t{}%".format(self.trueNegData, self.get_tn_ratio() * 100))
            print("False Positive:\t{},\t{}%".format(self.falsePosData, self.get_fp_ratio() * 100))
            print("False Negative:\t{},\t{}%".format(self.falseNegData, self.get_fn_ratio() * 100))
