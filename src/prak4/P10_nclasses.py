import copy
import random
import time

import numpy as np

from algorithms.decision_tree.decision_tree import DecisionTree
from algorithms.nearest_neighbors import NearestNeighbour
from algorithms.perceptron import Perceptron
from algorithms.transfers.signum import Signum
from metrics.euclidean import Euclidean
from misc.classification import Classification
from misc.feature_config import FeatureConfig
from misc.sensor import Sensor
from signals.windowed_signal import WindowedSignal
from stats.standard_deviation import StandardDeviation


laufen_classification = Classification(1.0, "Laufen")
gehen_classification = Classification(2.0, "Gehen")
huepfen_classification = Classification(3.0, "Huepfen")
drehen_classification = Classification(4.0, "Drehen")
ruhe_classification = Classification(5.0, "Ruhe")

enabled_classifications = [
    ruhe_classification,
    gehen_classification,
    huepfen_classification,
    drehen_classification,
    laufen_classification,
]

window_size = 500  # timestamps
window_moving = 200
stddev = StandardDeviation()
feature_configs = [FeatureConfig("accelX (m/s^2)", stddev, Sensor.get_sensor_ruecken()),
                   FeatureConfig("accelX (m/s^2)", stddev, Sensor.get_sensor_oberschenkel()),
                   FeatureConfig("accelX (m/s^2)", stddev, Sensor.get_sensor_unterschenkel()),
                   FeatureConfig("accelX (m/s^2)", stddev, Sensor.get_sensor_oberarm()),
                   FeatureConfig("accelX (m/s^2)", stddev, Sensor.get_sensor_unterarm())]

data_points = []
if ruhe_classification in enabled_classifications:
    ruhe_signal = WindowedSignal('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/ruhe.csv',
                             ruhe_classification, 0, 0, window_size, window_moving)
    data_points += ruhe_signal.get_data_points(feature_configs)
if gehen_classification in enabled_classifications:
    gehen_signal = WindowedSignal('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/gehen.csv',
                              gehen_classification, 5000, 5000, window_size, window_moving)
    data_points += gehen_signal.get_data_points(feature_configs)
if huepfen_classification in enabled_classifications:
    huepfen_signal = WindowedSignal('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/huepfen.csv',
                                huepfen_classification, 3000, 4500, window_size, window_moving)
    data_points += huepfen_signal.get_data_points(feature_configs)
if drehen_classification in enabled_classifications:
    drehen_signal = WindowedSignal('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/drehen.csv',
                               drehen_classification, 4000, 4500, window_size, window_moving)
    data_points += drehen_signal.get_data_points(feature_configs)
if laufen_classification in enabled_classifications:
    laufen_signal = WindowedSignal('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/laufen.csv',
                               laufen_classification, 4000, 4000, window_size, window_moving)
    data_points += laufen_signal.get_data_points(feature_configs)

random.shuffle(data_points)

teach_ratio = 0.4
teach_train_limit = int(np.round(teach_ratio * len(data_points)))

train_data_points = data_points[:teach_train_limit]
predict_data_points = data_points[teach_train_limit:]

for k in [3, 7, 11, 21, 51, 101]:
    start = time.time()
    knn = NearestNeighbour(k, Euclidean(), enabled_classifications)
    knn.train_data(train_data_points)
    for data_point in predict_data_points:
        knn.predict_data(data_point)
    end = time.time()
    print("\nKNN (k: {}): Duration: {}".format(k, end-start))
    knn.print_statistics()
for theta in [0.1, 0.01, 0.001, 0.0001]:
    for window_count in [5, 10, 20, 40, 80]:
        start = time.time()
        dt = DecisionTree(enabled_classifications, theta, window_count)
        dt.train_data(train_data_points)
        for data_point in predict_data_points:
            dt.predict_data(data_point)
        end = time.time()
        print("\nDecision-Tree (Theta: {}, Window-Count: {}): Duration {}".format(theta, window_count, end-start))
        dt.print_statistics()
