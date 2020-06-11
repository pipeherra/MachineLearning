import pandas as pd

from misc.sensor import Sensor
from stats.statistic import Statistic

data = pd.read_csv("../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/ruhe.csv")
sensors = Sensor.get_sensors()

for sensor in sensors:
    data_filtered = data[data['ID'] == sensor.id]
    print("{} (ID{})".format(sensor.name, sensor.id))

    columns = list(data_filtered)

    for column in columns:
        if column in ['ID', 'Timestamp']:
            continue
        values = data_filtered[column]
        print("\t{}: Mittelwert: {}, Median: {}, Standardabweichung: {}"
              .format(column, Statistic.get_mean(values),
                      Statistic.get_median(values),
                      Statistic.get_standard_deviation(values)))