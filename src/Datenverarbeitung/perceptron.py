import numpy as np
import matplotlib.pyplot as plt


class Perceptron:

    # tanh Transfger Function
    def activation(self, x):
        t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return t

    # Default constructor for a perceptron
    def __init__(self, maxEpochs, learning_rate, features):
        self.w = -1 + np.random.rand(features + 1) * 2  # Initialize weights randomly on range [-1:1]
        self.maxEpochs = maxEpochs  # Max iterations
        self.learning_rate = learning_rate  # Learning rate

    # Function to provide a guess for a specific input
    def guess(self, input):
        sum = 0
        for i in range(0, len(self.w)):
            sum += self.w[i] * input[i]
        return self.activation(sum)

    def trin(self):
        pass
