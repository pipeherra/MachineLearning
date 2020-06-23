import copy
import random
import time

import numpy as np
import xlsxwriter

from algorithms.decision_tree.decision_tree import DecisionTree
from algorithms.nearest_neighbors import NearestNeighbour
from algorithms.perceptron import Perceptron
from algorithms.transfers.signum import Signum
from metrics.chessboard import Chessboard
from metrics.euclidean import Euclidean
from metrics.manhattan import Manhattan
from misc.classification import Classification
from misc.feature_config import FeatureConfig
from misc.sensor import Sensor
from signals.windowed_signal import WindowedSignal
from stats.standard_deviation import StandardDeviation


def write_stat(ws, row, start_col, algorithm, duration):
    ws.write(row, start_col, algorithm.prediction_total_data_count)
    ws.write(row, start_col+1, algorithm.prediction_correct_data_count)
    ws.write(row, start_col+2, algorithm.get_correct_ratio())
    ws.write(row, start_col+3, algorithm.prediction_wrong_data_count)
    ws.write(row, start_col+4, algorithm.get_wrong_ratio())
    ws.write(row, start_col+5, algorithm.classifications[1].name)
    ws.write(row, start_col+6, algorithm.classifications[0].name)
    ws.write(row, start_col+7, algorithm.truePosData)
    ws.write(row, start_col+8, algorithm.get_tp_ratio())
    ws.write(row, start_col+9, algorithm.trueNegData)
    ws.write(row, start_col+10, algorithm.get_tn_ratio())
    ws.write(row, start_col+11, algorithm.falsePosData)
    ws.write(row, start_col+12, algorithm.get_fp_ratio())
    ws.write(row, start_col+13, algorithm.falseNegData)
    ws.write(row, start_col+14, algorithm.get_fn_ratio())
    ws.write(row, start_col+15, duration)


def write_header(ws, start_col):
    ws.write(0, start_col, "Predictions_total")
    ws.write(0, start_col+1, "Predictions_correct")
    ws.write(0, start_col+2, "Predictions_correct_rate")
    ws.write(0, start_col+3, "Predictions_wrong")
    ws.write(0, start_col+4, "Predictions_wrong_rate")
    ws.write(0, start_col+5, "Positive")
    ws.write(0, start_col+6, "Negative")
    ws.write(0, start_col+7, "true_positives")
    ws.write(0, start_col+8, "true_positives_rate")
    ws.write(0, start_col+9, "true_negatives")
    ws.write(0, start_col+10, "true_negatives_rate")
    ws.write(0, start_col+11, "false_positives")
    ws.write(0, start_col+12, "false_positives_rate")
    ws.write(0, start_col+13, "false_negatives")
    ws.write(0, start_col+14, "false_negatives_rate")
    ws.write(0, start_col+15, "Duration")


laufen_classification = Classification(1.0, "Laufen")
gehen_classification = Classification(1.0, "Gehen")
huepfen_classification = Classification(0.0, "Huepfen")
drehen_classification = Classification(0.0, "Drehen")
ruhe_classification = Classification(4.0, "Ruhe")

enabled_classifications = [
    #ruhe_classification,
    #gehen_classification,
    #huepfen_classification,
    drehen_classification,
    laufen_classification,
]

window_size = 250  # timestamps
window_moving = 100
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

workbook = xlsxwriter.Workbook("P10_2classes.xlsx")
ws_summary = workbook.add_worksheet("Summary")
ws_summary.write(0,  0, "Algorithm")
ws_summary.write(0,  1, "Parameter")
write_header(ws_summary, 2)
ws_summary_row = 1

ws_pla = workbook.add_worksheet("PLA")
ws_pla.write(0, 0, "Algorithm")
ws_pla.write(0, 1, "Learn-Rate")
ws_pla.write(0, 2, "Iterations")
write_header(ws_pla, 3)
ws_pla_row = 1

for learn_rate in [0.1, 0.01, 0.001]:
    for iterations in [10, 100, 1000, 10000, 100000]:
        start = time.time()
        pla = Perceptron(enabled_classifications, initial_weights, Signum(), False, False, learn_rate, iterations)
        pla.train_data(copy.deepcopy(train_data_points))
        for data_point in predict_data_points:
            pla.predict_data(data_point)
        end = time.time()
        print("PLA (Learn-Rate: {}, Iterations: {}): Success-Rate: {}"
              .format(learn_rate, iterations, pla.get_correct_ratio()))
        duration = end - start
        ws_summary.write(ws_summary_row, 0, "PLA")
        ws_summary.write(ws_summary_row, 1, "Learn-Rate: {}, Iterations: {}".format(learn_rate, iterations))
        write_stat(ws_summary, ws_summary_row, 2, pla, duration)
        ws_summary_row += 1
        ws_pla.write(ws_pla_row, 0, "PLA")
        ws_pla.write(ws_pla_row, 1, learn_rate)
        ws_pla.write(ws_pla_row, 2, iterations)
        write_stat(ws_pla, ws_pla_row, 3, pla, duration)
        ws_pla_row += 1

for learn_rate in [0.1, 0.01, 0.001]:
    for iterations in [10, 100, 1000, 10000, 100000]:
        start = time.time()
        pocket = Perceptron(enabled_classifications, initial_weights, Signum(), False, True, learn_rate, iterations)
        pocket.train_data(copy.deepcopy(train_data_points))
        for data_point in predict_data_points:
            pocket.predict_data(data_point)
        end = time.time()
        print("Pocket (Learn-Rate: {}, Iterations: {}): Success-Rate: {}"
              .format(learn_rate, iterations, pocket.get_correct_ratio()))
        duration = end - start
        ws_summary.write(ws_summary_row, 0, "Pocket")
        ws_summary.write(ws_summary_row, 1, "Learn-Rate: {}, Iterations: {}".format(learn_rate, iterations))
        write_stat(ws_summary, ws_summary_row, 2, pocket, duration)
        ws_summary_row += 1
        ws_pla.write(ws_pla_row, 0, "Pocket")
        ws_pla.write(ws_pla_row, 1, learn_rate)
        ws_pla.write(ws_pla_row, 2, iterations)
        write_stat(ws_pla, ws_pla_row, 3, pla, duration)
        ws_pla_row += 1

ws_knn = workbook.add_worksheet("KNN")
ws_knn.write(0, 0, "K")
ws_knn.write(0, 1, "Metric")
write_header(ws_knn, 2)
ws_knn_row = 1

for k in [3, 7, 11, 21, 51, 101]:
    for metric in [Chessboard(), Euclidean(), Manhattan()]:
        start = time.time()
        knn = NearestNeighbour(k, Euclidean(), enabled_classifications)
        knn.train_data(train_data_points)
        for data_point in predict_data_points:
            knn.predict_data(data_point)
        end = time.time()
        print("KNN (k: {}, Metric: {}): Success-Rate: {}".format(k, metric, knn.get_correct_ratio()))
        duration = end-start
        ws_summary.write(ws_summary_row, 0, "KNN")
        ws_summary.write(ws_summary_row, 1, "k: {}, metric: {}".format(k, metric))
        write_stat(ws_summary, ws_summary_row, 2, knn, duration)
        ws_summary_row += 1
        ws_knn.write(ws_knn_row, 0, k)
        ws_knn.write(ws_knn_row, 1, str(metric))
        write_stat(ws_knn, ws_knn_row, 1, knn, duration)
        ws_knn_row += 1

ws_dt = workbook.add_worksheet("Decision-Tree")
ws_dt.write(0, 0, "Theta")
ws_dt.write(0, 1, "window_size")
write_header(ws_dt, 2)
ws_dt_row = 1

for theta in [0.1, 0.01, 0.001, 0.0001]:
    for window_count in [5, 10, 20, 40, 80]:
        start = time.time()
        dt = DecisionTree(enabled_classifications, theta, window_count)
        dt.train_data(train_data_points)
        for data_point in predict_data_points:
            dt.predict_data(data_point)
        end = time.time()
        print("Decision-Tree (Theta: {}, Window-Count: {}): Success-Rate {}".format(theta, window_count, dt.get_correct_ratio()))
        duration = end-start
        ws_summary.write(ws_summary_row, 0, "Decision-Tree")
        ws_summary.write(ws_summary_row, 1, "Theta: {}, Window-Count: {}".format(theta, window_count))
        write_stat(ws_summary, ws_summary_row, 2, dt, duration)
        ws_summary_row += 1
        ws_dt.write(ws_dt_row, 0, theta)
        ws_dt.write(ws_dt_row, 1, window_count)
        write_stat(ws_dt, ws_dt_row, 2, dt, duration)
        ws_dt_row += 1

workbook.close()
