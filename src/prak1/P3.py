import pandas as pd
import matplotlib.pyplot as plt

from stats.statistic import Statistic

data = pd.read_csv('../../data/s0001.csv')
yvalues = data['f3']
xvalues = data['f2']

moving_average = Statistic.get_moving_average(yvalues, 3)
mean = Statistic.get_mean(yvalues)

plt.figure(figsize=(20, 10))
plt.scatter(xvalues, yvalues, label='f1')
# plt.scatter(moving_average.index, moving_average, label='moving_average[3]')
plt.hlines(mean, min(xvalues), max(xvalues), label="Mean")
plt.legend()
plt.show()
