import numpy as np


class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=10000, learning_rate=0.01):
        """
        :param no_of_inputs:  wie viele weights
        :param threshold: wie viele iterationen z.B 10000
        :param learning_rate: Rate für die Steps z.B 0,01
        """
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):

        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # w1*x1 + w2*x2 + w0
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        """
        :param training_inputs: inputs für predict()
        :param labels: Liste von Ergebnisse
        """
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
