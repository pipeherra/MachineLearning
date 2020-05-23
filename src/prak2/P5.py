import random

import pandas as pd
import matplotlib.pyplot as plt

from src.algorithmen.training_data import TrainingData
from src.algorithmen.perceptron import Perceptron
from src.misc.sensor import Sensor
from src.prak2.P5_data import P5Data
from src.signals.statistics import Statistics

threshold = 0.5

gehen_data = pd.read_csv('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/gehen.csv')
# walking starts not directly. Need to move the series
gehen_data = gehen_data[gehen_data.Timestamp >= (1492076265800 + 5000)]

ruhe_data = pd.read_csv('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/ruhe.csv')

# We do not need so many data -> clean up graphs
gehen_data['Timestamp_normalized'] = Statistics.get_timestamps_normalized(gehen_data['Timestamp'])
gehen_data = gehen_data[gehen_data.Timestamp_normalized < 22000]
ruhe_data['Timestamp_normalized'] = Statistics.get_timestamps_normalized(ruhe_data['Timestamp'])
ruhe_data = ruhe_data[ruhe_data.Timestamp_normalized < 22000]

plt.figure(figsize=(20, 10))
plt.scatter(ruhe_data['Timestamp_normalized'], ruhe_data['accelX (m/s^2)'], label='ruhe')
plt.scatter(gehen_data['Timestamp_normalized'], gehen_data['accelX (m/s^2)'], label='gehen')

data_array = []
sensors = Sensor.get_sensors()
features = 1 + len(sensors)

window_size = 500 #timestamps
moving_size = 200
window_count = int(21000/moving_size)

for i in range(window_count):
    start = i * moving_size
    end = i * moving_size + window_size
    gehen_range = gehen_data[(gehen_data.Timestamp_normalized >= start)
                             & (gehen_data.Timestamp_normalized <= end)]
    ruhe_range = ruhe_data[(ruhe_data.Timestamp_normalized >= start) & (ruhe_data.Timestamp_normalized <= end)]
    gehen_range_stddev = Statistics.get_standard_deviation(gehen_range['accelX (m/s^2)'])
    ruhe_range_stddev = Statistics.get_standard_deviation(ruhe_range['accelX (m/s^2)'])
    ruhe_features = [1.0, ruhe_range_stddev]
    gehen_features = [1.0, gehen_range_stddev]
    if features > 1:
        for sensor in sensors:
            ruhe_sensor = ruhe_range[ruhe_range['ID'] == sensor.id]
            gehen_sensor = gehen_range[gehen_range['ID'] == sensor.id]
            gehen_features.append(Statistics.get_standard_deviation(gehen_sensor['accelX (m/s^2)']))
            ruhe_features.append(Statistics.get_standard_deviation(ruhe_sensor['accelX (m/s^2)']))
    data_array.append(P5Data(i, start, end, ruhe_features, gehen_features))

perceptron = Perceptron.get_perceptron(threshold, features, 0.1)
random.shuffle(data_array)

training_data_array = []
prediction_data_array = []

for i in range(len(data_array)):
    data = data_array[i]
    if i == 0:
        plt.hlines(data.gehen_features[1], data.start, data.end, colors='orange', label='Stddev-Gehen')
        plt.hlines(data.ruhe_features[1], data.start, data.end, colors='silver', label='Stddev-Ruhe')
    else:
        plt.hlines(data.gehen_features[1], data.start, data.end, colors='orange')
        plt.hlines(data.ruhe_features[1], data.start, data.end, colors='silver')
    if i < int(window_count * 2 / 3):
        training_data_array.append(TrainingData(data.ruhe_features, 0.0))
        training_data_array.append(TrainingData(data.gehen_features, 1.0))
    else:
        prediction_data_array.append(TrainingData(data.ruhe_features, 0.0))
        prediction_data_array.append(TrainingData(data.gehen_features, 1.0))

perceptron.train_weight(training_data_array, True)

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0
for prediction_data in prediction_data_array:
    prediction = perceptron.predict(prediction_data.inputs)
    if prediction_data.expected == 1.0:
        if prediction == 1.0:
            true_positives += 1
        else:
            false_negatives += 1
    else:
        if prediction == 1.0:
            false_positives += 1
        else:
            true_negatives += 1
    # print("Predicting: Expected: {}, Predicted: {}".format(prediction_data.expected, prediction))
print("True-Positives: {}, True-Negatives: {}, False-Positives: {}, False-Negatives: {}"
      .format(true_positives, true_negatives, false_positives, false_negatives))
print("Error-Rate: {}, Success-Rate: {}".format((false_positives + false_negatives) / len(prediction_data_array),
                                                (true_positives + true_negatives) / len(prediction_data_array)))
plt.legend()
plt.show()
