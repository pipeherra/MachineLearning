from ML_tools import *

path = "data/18SoSe/2018_daten_grp5"

a = MovementDataGroup(path, ['gehen', 'joggen', 'rueckw', 'stehen', 'drehen'])

movements = ['gehen', 'joggen', 'rueckw', 'stehen']
sensors = ['gyroscop', 'accelerometer', 'magnet']

dimensions = [0, 1, 2]

position = 'upper arm'

for sensor in sensors:
    for dimension in dimensions:
        a.plot_scatter(sensor, position, dimension)

plt.show()