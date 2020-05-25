import src.algorithmen.nearest_neighbors as nn
import numpy as np
from statistics import *
import pandas as pd
import matplotlib.pyplot as plt

from src.misc.sensor import Sensor
from src.prak3.P8_data import P8Data
from src.signals.statistics import Statistics

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

window_size = 500 #timestamps
moving_size = 200
window_count = int(12000/moving_size)

for i in range(window_count):
    start = i * moving_size
    end = i * moving_size + window_size
    gehen_range = gehen_data[(gehen_data.Timestamp_normalized >= start)
                             & (gehen_data.Timestamp_normalized <= end)]
    ruhe_range = ruhe_data[(ruhe_data.Timestamp_normalized >= start) & (ruhe_data.Timestamp_normalized <= end)]
    huepfen_range = huepfen_data[(huepfen_data.Timestamp_normalized >= start) & (huepfen_data.Timestamp_normalized <= end)]
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
    data_array.append(P8Data(i, start, end, ruhe_features, gehen_features, huepfen_features))


k = 3
nearest_neighbours = nn.NearestNeighbour(k, 'euclidean', [1, 2, 3])

teach_ratio = 0.4
teach_train_limit = int(np.round(teach_ratio * len(data_array)))

# Train
for x in range(0, teach_train_limit):
    data = data_array[x]
    nearest_neighbours.train_data(data.gehen_features, 1)
    nearest_neighbours.train_data(data.ruhe_features, 2)
    nearest_neighbours.train_data(data.huepfen_features, 3)

# Test
for x in range(teach_train_limit, len(data_array)):
    data = data_array[x]
    nearest_neighbours.predict_data(data.gehen_features, 1)
    nearest_neighbours.predict_data(data.ruhe_features, 2)
    nearest_neighbours.predict_data(data.huepfen_features, 3)


# Print out
print("\n>>> NEAREST NEIGHBOURS <<<\n")
print("Tested Data: " + str(nearest_neighbours.testedData))
print("Correct Classifications: " + str(nearest_neighbours.correctClassifications))
print("Wrong Classifications: " + str(nearest_neighbours.wrongClassifications))

#plt.legend()
#plt.show()
