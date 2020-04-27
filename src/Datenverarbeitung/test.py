from src.signals.ploting import *
import csv
import numpy as np
import pandas

data_file = pandas.read_csv('gehen_P1.csv')

time = data_file['Timestamp']
time = time - time[0]

accel = [data_file['accelX (m/s^2)'], data_file['accelY (m/s^2)'], data_file['accelZ (m/s^2)']]

print(time[0:10])

# movements = ['gehen', 'joggen', 'rueckw', 'stehen']
# sensors = ['accelerometer', 'gyroscop', 'magnet']

# timestamps =
# dimensions = [0, 1, 2]
#
# position = 'upper arm'
#
# for sensor in sensors:
#     for dimension in dimensions:
#         scatter_plot(sensor, position, dimension)
#
# plt.show()
