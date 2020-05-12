import random

import pandas as pd
import matplotlib.pyplot as plt

from Datenverarbeitung.perceptron import Perceptron
from prak2.P5_data import P5Data
from src.signals.statistics import Statistics

threshold = 0.5
features = 1

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

for i in range(14):
    start = i * 1500
    end = (i + 1) * 1500
    gehen_range = gehen_data[(gehen_data.Timestamp_normalized >= start)
                             & (gehen_data.Timestamp_normalized <= end)]
    ruhe_range = ruhe_data[(ruhe_data.Timestamp_normalized >= start) & (ruhe_data.Timestamp_normalized <= end)]
    gehen_range_stddev = Statistics.get_standard_deviation(gehen_range['accelX (m/s^2)'])
    ruhe_range_stddev = Statistics.get_standard_deviation(ruhe_range['accelX (m/s^2)'])
    data_array.append(P5Data(i, start, end, ruhe_range_stddev, gehen_range_stddev))

perceptron = Perceptron.get_perceptron(threshold, features, 0.5)
random.shuffle(data_array)

for i in range(len(data_array)):
    data = data_array[i]
    if i == 0:
        plt.hlines(data.gehen_stddev, data.start, data.end, colors='orange', label='Stddev-Gehen')
        plt.hlines(data.ruhe_stddev, data.start, data.end, colors='silver', label='Stddev-Ruhe')
    else:
        plt.hlines(data.gehen_stddev, data.start, data.end, colors='orange')
        plt.hlines(data.ruhe_stddev, data.start, data.end, colors='silver')
    if i < 10:
        perceptron.train_weights([1.0, data.ruhe_stddev], 0.0)
        perceptron.train_weights([1.0, data.gehen_stddev], 1.0)
    else:
        print("Predicting Ruhe: Expected: {}, Predicted: {}"
              .format(0.0, perceptron.predict_with_normalized_tan([1.0, data.ruhe_stddev])))
        print("Predicting Gehen: Expected: {}, Predicted: {}"
              .format(1.0, perceptron.predict_with_normalized_tan([1.0, data.gehen_stddev])))
plt.legend()
plt.show()
