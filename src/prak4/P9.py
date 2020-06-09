import src.algorithmen.nearest_neighbors as nn
import numpy as np
from statistics import *
import pandas as pd
import matplotlib.pyplot as plt

from src.misc.sensor import Sensor
from src.prak4.P9_data import P9Data
from src.signals.statistics import Statistics


def unique_vals(rows, col):
    return set([row[col] for row in rows])


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


gehen_data = pd.read_csv('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/gehen.csv')
# walking starts not directly. Need to move the series
gehen_data = gehen_data[gehen_data.Timestamp >= (1492076265800 + 5000)]

huepfen_data = pd.read_csv('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/huepfen.csv')
huepfen_data = huepfen_data[huepfen_data.Timestamp >= (1492076761760 + 3000)]

ruhe_data = pd.read_csv('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/ruhe.csv')

# We do not need so many data -> clean up graphs
gehen_data['Timestamp_normalized'] = Statistics.get_timestamps_normalized(gehen_data['Timestamp'])
gehen_data = gehen_data[gehen_data.Timestamp_normalized < 12000]
ruhe_data['Timestamp_normalized'] = Statistics.get_timestamps_normalized(ruhe_data['Timestamp'])
ruhe_data = ruhe_data[ruhe_data.Timestamp_normalized < 12000]

huepfen_data['Timestamp_normalized'] = Statistics.get_timestamps_normalized(huepfen_data['Timestamp'])
huepfen_data = huepfen_data[huepfen_data.Timestamp_normalized < 12000]

plt.figure(figsize=(20, 10))
plt.scatter(ruhe_data['Timestamp_normalized'], ruhe_data['accelX (m/s^2)'], label='ruhe')
plt.scatter(gehen_data['Timestamp_normalized'], gehen_data['accelX (m/s^2)'], label='gehen')
plt.scatter(huepfen_data['Timestamp_normalized'], huepfen_data['accelX (m/s^2)'], label='huepfen')

data_array = []
sensors = Sensor.get_sensors()
features = 1 + len(sensors)

window_size = 500  # timestamps
moving_size = 200
window_count = int(12000 / moving_size)

for i in range(window_count):
    start = i * moving_size
    end = i * moving_size + window_size
    gehen_range = gehen_data[(gehen_data.Timestamp_normalized >= start)
                             & (gehen_data.Timestamp_normalized <= end)]
    ruhe_range = ruhe_data[(ruhe_data.Timestamp_normalized >= start) & (ruhe_data.Timestamp_normalized <= end)]
    huepfen_range = huepfen_data[
        (huepfen_data.Timestamp_normalized >= start) & (huepfen_data.Timestamp_normalized <= end)]
    gehen_range_stddev = Statistics.get_standard_deviation(gehen_range['accelX (m/s^2)'])
    ruhe_range_stddev = Statistics.get_standard_deviation(ruhe_range['accelX (m/s^2)'])
    huepfen_range_stddev = Statistics.get_standard_deviation(huepfen_range['accelX (m/s^2)'])
    ruhe_features = [ruhe_range_stddev]
    gehen_features = [gehen_range_stddev]
    huepfen_features = [huepfen_range_stddev]
    if features > 1:
        for sensor in sensors:
            ruhe_sensor = ruhe_range[ruhe_range['ID'] == sensor.id]
            gehen_sensor = gehen_range[gehen_range['ID'] == sensor.id]
            huepfen_sensor = huepfen_range[huepfen_range['ID'] == sensor.id]
            gehen_features.append(Statistics.get_standard_deviation(gehen_sensor['accelX (m/s^2)']))
            ruhe_features.append(Statistics.get_standard_deviation(ruhe_sensor['accelX (m/s^2)']))
            huepfen_features.append(Statistics.get_standard_deviation(huepfen_sensor['accelX (m/s^2)']))
    data_array.append(P9Data(i, start, end, ruhe_features, gehen_features, huepfen_features))

# Visualisierung
header = ["stdabw", "label"]


class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # Visualisierung
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Ist %s %s %s?" % (
            header[self.column], condition, str(self.value))


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    counts = 2  # Klassen
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain > best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    def __init__(self):
        self.predictions = 2


class Decision_Node:

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.question))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


if __name__ == '__main__':

    teach_ratio = 0.4
    teach_train_limit = int(np.round(teach_ratio * len(data_array)))

    # Train
    for x in range(0, teach_train_limit):
        data = data_array[x]
        my_tree = build_tree(data.gehen_features)
        my_tree = build_tree(data.ruhe_features)

    # Test
    detected = [1, 2, 3]
    for x in range(teach_train_limit, len(data_array)):
        data = data_array[x]
        print("Actual: %s. Predicted: %s" %
              (x[-1], print_leaf(classify(x, my_tree))))

    print_tree(my_tree)
