import math
import numpy as np


class Perceptron:

    def predict_sum(self, data_weights, data_inputs):
        threshold = data_weights.pop(0)
        if len(data_weights) != len(data_inputs):
            print("Len not equal")
            return
        data_sum = threshold
        for (data_weight, data_input) in zip(data_weights, data_inputs):
            data_sum += data_weight * data_input
        return data_sum

    def predict_with_normalized_tan(self, data_weights, data_inputs):
        data_sum = self.predict_sum(data_weights, data_inputs)
        return (math.tanh(data_sum) * 0.5) + 0.5

    def predict_sum_th(self, data_weights, data_inputs, threshold):
        if len(data_weights) != len(data_inputs):
            print("Len not equal")
            return
        data_sum = threshold
        for (data_weight, data_input) in zip(data_weights, data_inputs):
            data_sum += data_weight * data_input
        return data_sum

    def predict_with_normalized_tan_th(self, data_weights, data_inputs, threshold):
        data_sum = self.predict_sum_th(data_weights, data_inputs, threshold)
        return (math.tanh(data_sum) * 0.5) + 0.5

    def train(self, data_weights, data_inputs, data_class, learning_rate, threshold):
        iterations = 1000

        for iteration in range(1, iterations):

            prediction = self.predict_with_normalized_tan_th(data_weights, data_inputs, threshold)
            error = (data_class - prediction)

            if prediction != data_class:
                for i in range(1, len(data_weights)):
                    data_weights[i] += learning_rate * error * data_inputs[i]

        return data_weights
