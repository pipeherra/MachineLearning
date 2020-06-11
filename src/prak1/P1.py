import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# movements = ['walk', 'backwards_walk', 'steps', 'stand', 'run', 'jump', 'spin']

#PC
#data_file = pd.read_csv('C:/Users/Andres/meuml/data/17SoSe/2017_Gruppe1/huepfen_P1.csv')

# Mac
data_file = pd.read_csv('../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/ruhe.csv')

#data_file = pd.read_csv('C:/Users/Andres/meuml/src/Datenverarbeitung/s0001.csv')

time = []
sensor_data = []
temp_data_file = []

part = 'lower arm'

if part == 'lower arm':
    temp_data_file = data_file.loc[data_file['ID'] == 1324180]
elif part == 'upper arm':
    temp_data_file = data_file.loc[data_file['ID'] == 1324185]
elif part == 'upper leg':
    temp_data_file = data_file.loc[data_file['ID'] == 1324184]
elif part == 'lower leg':
    temp_data_file = data_file.loc[data_file['ID'] == 1324187]
elif part == 'back':
    temp_data_file = data_file.loc[data_file['ID'] == 1324186]

time = temp_data_file['Timestamp']
#time = time - time[0]

sensor_data = temp_data_file.iloc[:, 2:11]

sensors = {
    'acce': {'columns': [0, 1, 2],
             'color': 'r',
             'group': 'Accelerometer'
             },
    'gyro': {'columns': [3, 4, 5],
             'color': "g",
             'group': 'Gyroscope'
             },
    'mag': {'columns': [6, 7, 8],
            'color': 'b',
            'group': 'Magnet'
            }
}

for key, val in sensors.items():
    if key is 'acce':
        column = val['columns']
        x1 = time
        x2 = time
        x3 = time
        y1 = sensor_data.iloc[:, column[0]]
        y2 = sensor_data.iloc[:, column[1]]
        y3 = sensor_data.iloc[:, column[2]]

        plt.scatter(time, y1, alpha=0.8, c=val['color'], edgecolors='none', s=30, label=val['group'], marker='x')
        #plt.scatter(time, y2, alpha=0.8, c=val['color'], edgecolors='none', s=30, label=val['group'], marker='1')
        #plt.scatter(time, y3, alpha=0.8, c=val['color'], edgecolors='none', s=30, label=val['group'], marker='+')

plt.show()
