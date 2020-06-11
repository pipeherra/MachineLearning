from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from misc.classification import Classification
from misc.data_point import DataPoint
from metrics.euclidean import Euclidean
from algorithms.nearest_neighbors import NearestNeighbour
from src.misc.sensor import Sensor
from src.prak3.P7_data import P7Data
from src.signals.statistics import Statistics

gehen_data = pd.read_csv('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/gehen.csv')
# walking starts not directly. Need to move the series
gehen_data = gehen_data[gehen_data.Timestamp >= (1492076265800 + 5000)]

ruhe_data = pd.read_csv('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/ruhe.csv')

# We do not need so many data -> clean up graphs
gehen_data['Timestamp_normalized'] = Statistics.get_timestamps_normalized(gehen_data['Timestamp'])
gehen_data = gehen_data[gehen_data.Timestamp_normalized < 22000]
ruhe_data['Timestamp_normalized'] = Statistics.get_timestamps_normalized(ruhe_data['Timestamp'])
ruhe_data = ruhe_data[ruhe_data.Timestamp_normalized < 22000]

ruhe_classification = Classification(0.0, "Ruhe")
gehen_classification = Classification(1.0, "Gehen")

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
    data_array.append(P7Data(i, start, end, ruhe_features, gehen_features))


k = 3
nearest_neighbours = NearestNeighbour(k, Euclidean(), [ruhe_classification, gehen_classification])

teach_ratio = 0.4
teach_train_limit = int(np.round(teach_ratio * len(data_array)))

# Train
train_data: List[DataPoint] = []
for x in range(0, teach_train_limit):
    data = data_array[x]
    train_data.append(DataPoint(data.gehen_features, gehen_classification))
    train_data.append(DataPoint(data.ruhe_features, ruhe_classification))
nearest_neighbours.train_data(train_data)

# Test
for x in range(teach_train_limit, len(data_array)):
    data = data_array[x]
    nearest_neighbours.predict_data(DataPoint(data.gehen_features, gehen_classification))
    nearest_neighbours.predict_data(DataPoint(data.ruhe_features, ruhe_classification))

print("\n>>> NEAREST NEIGHBOURS <<<\n")
nearest_neighbours.print_statistics()
