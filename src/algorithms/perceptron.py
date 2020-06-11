from typing import List

import numpy as np

from algorithms.algorithm import Algorithm
from misc.classification import Classification
from misc.data_point import DataPoint
from algorithms.transfers.transfer import Transfer


class Perceptron(Algorithm):
    def __init__(self, classifications: List[Classification], initial_weights, transfer: Transfer,
                 debug=False, pocket=False, learning_rate=0.01, max_iterations=10000):
        if len(classifications) != 2:
            raise AttributeError("Perceptron is only able to differ two classes!")
        super().__init__(classifications)
        self.weights = initial_weights
        self.transfer = transfer
        self.debug = debug
        self.pocket = pocket
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def train_data(self, data_array: List[DataPoint]):
        updates = 0
        wrong_predictions = np.inf
        last_wrong_predictions = wrong_predictions
        weights = self.weights.copy()
        for i in range(len(data_array)):
            data_array[i].features = [1.0] + data_array[i].features
        while wrong_predictions != 0 and updates < self.max_iterations:
            wrong_predictions = 0
            for training_data in data_array:
                prediction = self.test_weights(weights, training_data.features)
                error = training_data.class_expected.value - prediction
                if error != 0.0:
                    updates += 1
                    for i in range(len(self.weights)):
                        weights[i] += self.learning_rate * error * training_data.features[i]
                    break
            for training_data in data_array:
                prediction = self.test_weights(weights, training_data.features)
                if prediction != training_data.class_expected.value:
                    wrong_predictions += 1
            if self.pocket:
                if wrong_predictions < last_wrong_predictions:
                    self.weights = weights.copy()
                    last_wrong_predictions = wrong_predictions
                    if self.debug:
                        print(
                            "Updated Weights: Data_Count: {}, Wrong_Predictions: {}, Error-Rate: {}, Total-Updates: {}"
                                .format(len(data_array), wrong_predictions,
                                        wrong_predictions / len(data_array), updates))
                elif self.debug:
                    print("Discarded Weights: Data_Count: {}, Wrong_Predictions: {}, Error-Rate: {}, Total-Updates: {}"
                          .format(len(data_array), wrong_predictions,
                                  wrong_predictions / len(data_array), updates))
            else:
                if self.debug:
                    print("Data_Count: {}, Wrong_Predictions: {}, Error-Rate: {}, Total-Updates: {}"
                          .format(len(data_array), wrong_predictions,
                                  wrong_predictions / len(data_array), updates))
        if not self.pocket:
            self.weights = weights

    def predict_data(self, data_point: DataPoint) -> Classification:
        prediction_value = self.test_weights(self.weights, [1.0] + data_point.features)
        prediction_class = self.to_classification(prediction_value)
        self.update_rates(data_point.class_expected, prediction_class)
        return prediction_class

    def test_weights(self, test_weights, data_inputs) -> float:
        if len(test_weights) != len(data_inputs):
            raise AttributeError("Len not equal!")
        data_sum = 0.0
        for (data_weight, data_input) in zip(test_weights, data_inputs):
            data_sum += data_weight * data_input
        return self.transfer.transfer(data_sum)

    def update_weights(self, data_point: DataPoint):
        prediction = self.test_weights(self.weights, data_point.features)
        error = data_point.class_expected.value - prediction
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * data_point.features[i]
        return self.weights
