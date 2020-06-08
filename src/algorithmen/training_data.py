import csv
import matplotlib.pyplot as plt


class TrainingData:
    def __init__(self, inputs, expected):
        self.inputs = inputs
        self.expected = expected
