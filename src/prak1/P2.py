import pandas as pd

from stats.statistic import Statistic

data = pd.read_csv('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/gehen.csv')
print("Gehen:")

columns = list(data)

for column in columns:
    if column in ['ID', 'Timestamp']:
        continue
    values = data[column].to_numpy()
    print("{}: Mittelwert: {}, Median: {}, Standardabweichung: {}".format(column, Statistic.get_mean(values),
          Statistic.get_median(values), Statistic.get_standard_deviation(values)))

data = pd.read_csv('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/ruhe.csv')
print("\nRuhe:")
columns = list(data)

for column in columns:
    if column in ['ID', 'Timestamp']:
        continue
    values = data[column].to_numpy()
    print("{}: Mittelwert: {}, Median: {}, Standardabweichung: {}".format(column, Statistic.get_mean(values),
          Statistic.get_median(values), Statistic.get_standard_deviation(values)))
