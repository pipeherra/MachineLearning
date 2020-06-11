import random

import numpy as np

from algorithms.decision_tree.decision_tree import DecisionTree
from algorithms.nearest_neighbors import NearestNeighbour
from algorithms.perceptron import Perceptron
from algorithms.transfers.signum import Signum
from algorithms.transfers.tanh import Tanh
from metrics.euclidean import Euclidean
from misc.classification import Classification
from misc.feature_config import FeatureConfig
from signals.windowed_signal import WindowedSignal
from src.misc.sensor import Sensor
from stats.standard_deviation import StandardDeviation

ruhe_classification = Classification(0.0, "Ruhe")
gehen_classification = Classification(1.0, "Gehen")
huepfen_classification = Classification(2.0, "Huepfen")
drehen_classification = Classification(3.0, "Drehen")
laufen_classification = Classification(4.0, "Laufen")

enabled_classifications = [
    #ruhe_classification,
    #gehen_classification,
    huepfen_classification,
    #drehen_classification,
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

initial_weights = np.zeros(len(feature_configs) + 1)
initial_weights[0] = 0.5
#algorithm = Perceptron(enabled_classifications, initial_weights, Signum(), True, False, 0.01, 10000)
#algorithm = NearestNeighbour(3, Euclidean(), enabled_classifications)
algorithm = DecisionTree([ruhe_classification, gehen_classification], 0.05, 10)

algorithm.train_data(train_data_points)

#for predict_data_point in predict_data_points:
#    algorithm.predict_data(predict_data_point)

#algorithm.print_statistics()
algorithm.print_tree()
print("done")
