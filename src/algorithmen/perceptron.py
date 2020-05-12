import math
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

    def update_weights(self, learning_data):
        if len(learning_data.inputs) != len(self.weights):
            raise AttributeError('Len not equal')
        prediction = self.predict(learning_data.inputs)
        error = learning_data.expected - prediction
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * learning_data.inputs[i]
        return self.weights

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

