import pandas as pd

from src.signals.statistics import Statistics

data = pd.read_csv('../../data/19SoSe/2019_daten_grp2/ruhelage.csv')

columns = list(data)

for column in columns:
    if column in ['ID', 'Timestamp']:
        continue
    values = data[column].to_numpy()
    print("{}: Mittelwert: {}, Median: {}, Standardabweichung: {}".format(column, Statistics.get_mean(values),
          Statistics.get_median(values), Statistics.get_standard_deviation(values)))
