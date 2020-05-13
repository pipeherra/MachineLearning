import math
import random

import numpy as np


class Perceptron:
    def __init__(self, transfer_function, initial_weights, learning_rate=0.0):
        self.transfer_function = transfer_function
        self.weights = initial_weights
        self.learning_rate = learning_rate

    def predict(self, data_inputs):
        if len(self.weights) != len(data_inputs):
            print("Len not equal")
            return
        data_sum = 0.0
        for (data_weight, data_input) in zip(self.weights, data_inputs):
            data_sum += data_weight * data_input
        return self.transfer_function(data_sum)

    def test_weights(self, data_inputs, test_weights):
        if len(self.weights) != len(data_inputs):
            print("Len not equal")
            return
        data_sum = 0.0
        for (data_weight, data_input) in zip(test_weights, data_inputs):
            data_sum += data_weight * data_input
        return self.transfer_function(data_sum)

    def update_weights(self, learning_data):
        if len(learning_data.inputs) != len(self.weights):
            raise AttributeError('Len not equal')
        prediction = self.predict(learning_data.inputs)
        error = learning_data.expected - prediction
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * learning_data.inputs[i]
        return self.weights

    def train_weight(self, training_data_array, pocket):
        updates = 0
        wrong_predictions = np.inf
        last_wrong_predictions = wrong_predictions
        weights = self.weights.copy()
        while wrong_predictions != 0 and updates < 100000:
            wrong_predictions = 0
            for training_data in training_data_array:
                prediction = self.test_weights(training_data.inputs, weights)
                error = training_data.expected - prediction
                if error != 0.0:
                    updates += 1
                    for i in range(len(self.weights)):
                        weights[i] += self.learning_rate * error * training_data.inputs[i]
                    break
            for training_data in training_data_array:
                prediction = self.test_weights(training_data.inputs, weights)
                if prediction != training_data.expected:
                    wrong_predictions += 1
            if pocket:
                if wrong_predictions < last_wrong_predictions:
                    self.weights = weights.copy()
                    last_wrong_predictions = wrong_predictions
                    print("Updated Weights: Data_Count: {}, Wrong_Predictions: {}, Error-Rate: {}, Total-Updates: {}"
                          .format(len(training_data_array), wrong_predictions,
                                  wrong_predictions/len(training_data_array), updates))
                else:
                    print("Discarded Weights: Data_Count: {}, Wrong_Predictions: {}, Error-Rate: {}, Total-Updates: {}"
                          .format(len(training_data_array), wrong_predictions,
                                  wrong_predictions / len(training_data_array), updates))
            else:
                print("Data_Count: {}, Wrong_Predictions: {}, Error-Rate: {}, Total-Updates: {}"
                      .format(len(training_data_array), wrong_predictions,
                              wrong_predictions / len(training_data_array), updates))


    @staticmethod
    def normalized_tanh(data_sum):
        return (math.tanh(data_sum) * 0.5) + 0.5

    @staticmethod
    def signum(data_sum):
        if data_sum < 0.0:
            return 0.0
        return 1.0

    @staticmethod
    def get_perceptron(threshold, features, learning_rate):
        initial_weights = np.zeros(features + 1)
        initial_weights[0] = threshold
        return Perceptron(Perceptron.signum, initial_weights, learning_rate)

