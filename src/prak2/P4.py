import pandas as pd
import matplotlib.pyplot as plt


gehen_data = pd.read_csv('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/gehen.csv')
ruhe_data = pd.read_csv('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/ruhe.csv')

gehen_data['Timestamp_normalized'] = Statistics.get_timestamps_normalized(gehen_data['Timestamp'])
ruhe_data['Timestamp_normalized'] = Statistics.get_timestamps_normalized(ruhe_data['Timestamp'])

gehen_filtered = gehen_data[gehen_data['Timestamp_normalized'] <= 15000]
ruhe_filtered = ruhe_data[ruhe_data['Timestamp_normalized'] <= 15000]

plt.figure(figsize=(20, 10))
plt.scatter(ruhe_filtered['Timestamp_normalized'], ruhe_filtered['accelX (m/s^2)'], label='ruhe')
plt.scatter(gehen_filtered['Timestamp_normalized'], gehen_filtered['accelX (m/s^2)'], label='gehen')
plt.hlines(Statistics.get_standard_deviation(ruhe_filtered['accelX (m/s^2)']), 0, 15000, label="Standardabweichung Ruhe")
plt.hlines(Statistics.get_standard_deviation(gehen_filtered['accelX (m/s^2)']), 0, 15000, label="Standardabweichung Gehen")
plt.legend()
plt.show()
