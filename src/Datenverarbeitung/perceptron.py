import math


class Perceptron:
    # Default constructor for a perceptron
    # def __init__(self):
    # self.w = -1 + np.random.rand(features + 1) * 2  # Initialize weights randomly on range [-1:1]
    # self.maxEpochs = maxEpochs  # Max iterations
    # self.learning_rate = learning_rate  # Learning rate

    # def trin(self):
    #    pass

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
