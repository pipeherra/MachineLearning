import pandas as pd

from Datenverarbeitung.sensor import Sensor
from signals.statistics import Statistics

data = pd.read_csv("../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/ruhe.csv")
sensors = Sensor.get_sensors()

for sensor in sensors:
    data_filtered = data[data['ID'] == sensor.id]
    print("{} (ID{})".format(sensor.name, sensor.id))

    columns = list(data_filtered)

    for column in columns:
        if column in ['ID', 'Timestamp']:
            continue
        values = data[column]
        print("\t{}: Mittelwert: {}, Median: {}, Standardabweichung: {}"
              .format(column, Statistics.get_mean(values),
                      Statistics.get_median(values),
                      Statistics.get_standard_deviation(values)))
