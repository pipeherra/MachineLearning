import math


class Perceptron:
    def __init__(self, initial_weights, learning_rate=0.0):
        self.weights = initial_weights
        self.learning_rate = learning_rate

    def predict_sum(self, data_inputs):
        if len(self.weights) != len(data_inputs):
            print("Len not equal")
            return
        data_sum = 0.0
        for (data_weight, data_input) in zip(self.weights, data_inputs):
            data_sum += data_weight * data_input
        return data_sum

    def predict_with_normalized_tan(self, data_inputs):
        data_sum = self.predict_sum(data_inputs)
        return (math.tanh(data_sum) * 0.5) + 0.5

    def train_weights(self, data_inputs, data_expected):
        if len(data_inputs) != len(self.weights):
            print("len not equal")
            return
        prediction = self.predict_with_normalized_tan(data_inputs)
        error = data_expected - prediction
        for i in range(len(data_inputs)):
            self.weights[i] += self.learning_rate * error * data_inputs[i]
        return self.weights

