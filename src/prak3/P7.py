import src.algorithmen.nearest_neighbors as nn
import numpy as np
from statistics import *
import pandas as pd
import matplotlib.pyplot as plt

from src.misc.sensor import Sensor
from src.prak3.P7_data import P7Data
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
    data_array.append(P7Data(i, start, end, ruhe_features, gehen_features))


k = 3
nearest_neighbours = nn.NearestNeighbour(k, 'euclidean')

teach_ratio = 0.4
teach_train_limit = int(np.round(teach_ratio * len(data_array)))

# Train
for x in range(0, teach_train_limit):
    data = data_array[x]
    nearest_neighbours.train_data(data.gehen_features, 1)
    nearest_neighbours.train_data(data.ruhe_features, 0)

# Test
for x in range(teach_train_limit, len(data_array)):
    data = data_array[x]
    nearest_neighbours.predict_data(data.gehen_features, 1)
    nearest_neighbours.predict_data(data.ruhe_features, 0)


# for i in range(len(data_array)):
#     data = data_array[i]
#     if i == 0:
#         plt.hlines(data.gehen_features[1], data.start, data.end, colors='orange', label='Stddev-Gehen')
#         plt.hlines(data.ruhe_features[1], data.start, data.end, colors='silver', label='Stddev-Ruhe')
#     else:
#         plt.hlines(data.gehen_features[1], data.start, data.end, colors='orange')
#         plt.hlines(data.ruhe_features[1], data.start, data.end, colors='silver')
#     if i < int(window_count * 2 / 3):
#         nearest_neighbours.train_data(data.ruhe_features, 0.0)
#         nearest_neighbours.train_data(data.gehen_features, 1.0)
#     else:
#         nearest_neighbours.test_data(data.ruhe_features, 0.0)
#         nearest_neighbours.test_data(data.gehen_features, 1.0)

# Print out
print("\n>>> NEAREST NEIGHBOURS <<<\n")
print("Tested Data: " + str(nearest_neighbours.testedData))
print("True Positive Data: " + str(nearest_neighbours.truePosData))
print("True Negative Data: " + str(nearest_neighbours.trueNegData))
print("False Positive Data: " + str(nearest_neighbours.falsePosData))
print("False Negative Data: " + str(nearest_neighbours.falseNegData))

print("Fehler Rate: " + str(nearest_neighbours.get_f_ratio()))
print("True Positive Rate: " + str(nearest_neighbours.get_tp_ratio()))
print("True Negative Rate: " + str(nearest_neighbours.get_tn_ratio()))
print("False Positive Rate: " + str(nearest_neighbours.get_fp_ratio()))
print("False Negative Rate: " + str(nearest_neighbours.get_fn_ratio()))
